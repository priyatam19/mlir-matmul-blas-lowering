#!/usr/bin/env python3
"""Fused linear -> GELU Triton kernel.

Mirrors the up-projection of a GPT-2-style FFN block at this repo's own
benchmarked shape (`src/benchmarks/gpt2_ffn.mlir`, also used by
`benchmarks/benchmark_shapes.py`'s "gpt2_ffn" config: M=128, K=256, N=512),
extended with a bias term and GELU to match the actual fusable unit of a
real transformer FFN -- the checked-in `.mlir` benchmarks the bare matmul
only. This is also exactly the op
`docs/tensor_parallel_placement_analysis.md` column-shards across GPUs: that
doc reasons about where the compute/communication boundary falls; this
kernel is what runs inside one shard.

GELU uses the exact erf-based formula (matching
`torch.nn.functional.gelu`'s default, not the tanh approximation), since
`triton.language.erf` is available directly and there's no reason to trade
accuracy for an approximation the platform doesn't need.

Correctness doesn't require a GPU: `--interpret` runs the same kernel code
through Triton's CPU interpreter (set `TRITON_INTERPRET=1`) and checks it
against `torch.nn.functional.gelu`. Real timing needs a CUDA GPU.
"""

from __future__ import annotations

import argparse
import os

from matmul import benchmark
from triton_utils import print_timing, require_cuda, require_torch_and_triton

torch, triton, tl = require_torch_and_triton()

_INV_SQRT2 = 0.7071067811865476


@triton.jit
def _fused_linear_gelu_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k_idxs = k_start + offs_k
        x_mask = (offs_m[:, None] < M) & (k_idxs[None, :] < K)
        x_tile = tl.load(x_ptr + offs_m[:, None] * stride_xm + k_idxs[None, :] * stride_xk,
                          mask=x_mask, other=0.0)
        w_mask = (k_idxs[:, None] < K) & (offs_n[None, :] < N)
        w_tile = tl.load(w_ptr + k_idxs[:, None] * stride_wk + offs_n[None, :] * stride_wn,
                          mask=w_mask, other=0.0)
        acc += tl.dot(x_tile, w_tile)

    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    pre_act = acc + bias[None, :]
    # exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    result = 0.5 * pre_act * (1.0 + tl.erf(pre_act * _INV_SQRT2))

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
             result, mask=out_mask)


# See fused_linear_clamp.py's identical note: autotune wraps the raw kernel
# rather than decorating it, so `--interpret` can launch the same kernel body
# with a fixed config without going through autotune's driver-dependent
# benchmarking harness.
_fused_linear_gelu_autotuned = triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8),
    ],
    key=["M", "N", "K"],
)(_fused_linear_gelu_kernel)


def fused_linear_gelu(x, weight, bias):
    """gelu(x @ weight + bias). `weight` is (in_features, out_features)."""
    M, K = x.shape
    K2, N = weight.shape
    if K != K2:
        raise ValueError(f"incompatible shapes: x={tuple(x.shape)} weight={tuple(weight.shape)}")
    out = torch.empty((M, N), device=x.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
    _fused_linear_gelu_autotuned[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
    )
    return out


def _fused_linear_gelu_fixed_config(x, weight, bias, block_m=32, block_n=32, block_k=32):
    """Bypasses autotune -- used by `--interpret`."""
    M, K = x.shape
    K2, N = weight.shape
    if K != K2:
        raise ValueError(f"incompatible shapes: x={tuple(x.shape)} weight={tuple(weight.shape)}")
    out = torch.empty((M, N), device=x.device, dtype=torch.float32)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    _fused_linear_gelu_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
    )
    return out


def unfused_reference(x, weight, bias):
    return torch.nn.functional.gelu(x @ weight + bias)


def _run_interpreted(m, k, n):
    if os.environ.get("TRITON_INTERPRET") != "1":
        raise SystemExit("--interpret requires TRITON_INTERPRET=1 in the environment")
    torch.manual_seed(41)
    x = torch.randn(m, k)
    weight = torch.randn(k, n)
    bias = torch.randn(n)
    actual = _fused_linear_gelu_fixed_config(x, weight, bias)
    expected = unfused_reference(x, weight, bias)
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-4)
    print(f"interpreted (no GPU): fused_linear_gelu matches unfused reference, shape M={m} K={k} N={n}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--interpret", action="store_true",
                         help="Check correctness on CPU via Triton's interpreter; needs TRITON_INTERPRET=1. No timing.")
    args = parser.parse_args()

    if args.interpret:
        _run_interpreted(args.m, args.k, args.n)
        return

    require_cuda(torch)
    torch.manual_seed(41)

    x = torch.randn((args.m, args.k), device="cuda", dtype=torch.float32)
    weight = torch.randn((args.k, args.n), device="cuda", dtype=torch.float32)
    bias = torch.randn((args.n,), device="cuda", dtype=torch.float32)

    actual = fused_linear_gelu(x, weight, bias)
    expected = unfused_reference(x, weight, bias)
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-4)

    print(f"fused_linear_gelu shape: M={args.m} K={args.k} N={args.n} (gpt2_ffn up-projection)")
    print_timing("triton fused", benchmark(lambda: fused_linear_gelu(x, weight, bias), args.warmup, args.runs))
    print_timing("torch unfused", benchmark(lambda: unfused_reference(x, weight, bias), args.warmup, args.runs))


if __name__ == "__main__":
    main()
