#!/usr/bin/env python3
"""Fused (x + param) -> linear -> clamp Triton kernel.

Mirrors `src/sample/model.py`'s `Sample.forward`:

    clamp(linear(x + self.param, self.linear.weight, self.linear.bias), 0, 1)

This is the exact subgraph `src/torch_compile/mlir_backend.py` accepts and
lowers through the MLIR pipeline instead of Triton, and the extension
`docs/triton_first_roadmap.md` named as the near-term step after the
vector_add/matmul smoke tests. It gives a fourth point of comparison for
`src/torch_compile/demo.py`: eager, default Inductor, this project's own
MLIR backend, and a hand-written fused Triton kernel, all computing the same
function.

Fusion: the elementwise add, the GEMM, the bias broadcast-add, and the
clamp all happen in one kernel launch and one pass through registers -- the
intermediates (`x + param`, the pre-clamp linear output) never round-trip
through global memory the way three separate torch ops would.

Correctness doesn't require a GPU: `--interpret` runs the same kernel code
through Triton's CPU interpreter (set `TRITON_INTERPRET=1`) and checks it
against `unfused_reference`. Real timing needs a CUDA GPU, same as this
directory's other scripts.
"""

from __future__ import annotations

import argparse
import os

from matmul import benchmark
from triton_utils import print_timing, require_cuda, require_torch_and_triton

torch, triton, tl = require_torch_and_triton()


@triton.jit
def _fused_linear_clamp_kernel(
    x_ptr, param_ptr, w_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_pm, stride_pk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    clamp_min: tl.constexpr, clamp_max: tl.constexpr,
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
        a_mask = (offs_m[:, None] < M) & (k_idxs[None, :] < K)
        x_tile = tl.load(x_ptr + offs_m[:, None] * stride_xm + k_idxs[None, :] * stride_xk,
                          mask=a_mask, other=0.0)
        p_tile = tl.load(param_ptr + offs_m[:, None] * stride_pm + k_idxs[None, :] * stride_pk,
                          mask=a_mask, other=0.0)
        a_tile = x_tile + p_tile  # the fused elementwise add

        w_mask = (offs_n[:, None] < N) & (k_idxs[None, :] < K)
        # nn.Linear stores weight as (out_features, in_features) = (N, K); load transposed.
        w_tile = tl.load(w_ptr + offs_n[:, None] * stride_wn + k_idxs[None, :] * stride_wk,
                          mask=w_mask, other=0.0)
        acc += tl.dot(a_tile, tl.trans(w_tile))

    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    result = acc + bias[None, :]
    result = tl.minimum(tl.maximum(result, clamp_min), clamp_max)

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
             result, mask=out_mask)


# Autotune wraps the raw kernel above rather than decorating it directly, so
# `--interpret` (see `_fused_linear_clamp_fixed_config` below) can launch the
# same kernel body with an explicit config -- triton.autotune's own
# benchmarking harness calls into the active GPU driver even when kernel
# *execution* is running under TRITON_INTERPRET's CPU interpreter, so
# autotuning itself is not something interpreter mode can exercise.
_fused_linear_clamp_autotuned = triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 16}, num_warps=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 16}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8),
    ],
    key=["M", "N", "K"],
)(_fused_linear_clamp_kernel)


def fused_linear_clamp(x, param, weight, bias, clamp_min=0.0, clamp_max=1.0):
    """clamp((x + param) @ weight.T + bias, clamp_min, clamp_max)."""
    M, K = x.shape
    N, K2 = weight.shape
    if K != K2:
        raise ValueError(f"incompatible shapes: x={tuple(x.shape)} weight={tuple(weight.shape)}")
    out = torch.empty((M, N), device=x.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
    _fused_linear_clamp_autotuned[grid](
        x, param, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        param.stride(0), param.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        clamp_min, clamp_max,
    )
    return out


def _fused_linear_clamp_fixed_config(x, param, weight, bias, clamp_min=0.0, clamp_max=1.0,
                                      block_m=16, block_n=16, block_k=16):
    """Bypasses autotune -- used by `--interpret`, which has no driver for
    autotune's own benchmarking to run against (see note above)."""
    M, K = x.shape
    N, K2 = weight.shape
    if K != K2:
        raise ValueError(f"incompatible shapes: x={tuple(x.shape)} weight={tuple(weight.shape)}")
    out = torch.empty((M, N), device=x.device, dtype=torch.float32)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    _fused_linear_clamp_kernel[grid](
        x, param, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        param.stride(0), param.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        clamp_min, clamp_max,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
    )
    return out


def unfused_reference(x, param, weight, bias, clamp_min=0.0, clamp_max=1.0):
    """Same computation via three separate torch ops -- the fusion baseline."""
    return torch.clamp(torch.nn.functional.linear(x + param, weight, bias), clamp_min, clamp_max)


def _run_interpreted(m, k, n):
    if os.environ.get("TRITON_INTERPRET") != "1":
        raise SystemExit("--interpret requires TRITON_INTERPRET=1 in the environment")
    torch.manual_seed(41)
    x = torch.randn(m, k)
    param = torch.randn(m, k)
    weight = torch.randn(n, k)
    bias = torch.randn(n)
    actual = _fused_linear_clamp_fixed_config(x, param, weight, bias)
    expected = unfused_reference(x, param, weight, bias)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    print(f"interpreted (no GPU): fused_linear_clamp matches unfused reference, shape M={m} K={k} N={n}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=3)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--n", type=int, default=5)
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
    param = torch.randn((args.m, args.k), device="cuda", dtype=torch.float32)
    weight = torch.randn((args.n, args.k), device="cuda", dtype=torch.float32)
    bias = torch.randn((args.n,), device="cuda", dtype=torch.float32)

    actual = fused_linear_clamp(x, param, weight, bias)
    expected = unfused_reference(x, param, weight, bias)
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-4)

    print(f"fused_linear_clamp shape: M={args.m} K={args.k} N={args.n} (src/sample/model.py's Sample)")
    print_timing("triton fused", benchmark(lambda: fused_linear_clamp(x, param, weight, bias), args.warmup, args.runs))
    print_timing("torch unfused", benchmark(lambda: unfused_reference(x, param, weight, bias), args.warmup, args.runs))


if __name__ == "__main__":
    main()
