#!/usr/bin/env python3
"""Analytical tensor-parallel placement study for a GPT-2-style FFN block.

Extends this project's own roofline methodology
(docs/deep_dive_part3_qa6to10.md, Q76/Q81) one level up the memory
hierarchy: instead of asking whether a kernel is compute- or
DRAM-bandwidth-bound on one chip, this asks whether a Megatron-style
tensor-parallel MLP split is compute- or interconnect-bound across
multiple L4s -- the actual GPU this project targets, which has no NVLink,
so the interconnect is PCIe or nothing.

No multi-GPU hardware is required: this is a closed-form cost model, the
same kind of estimate lib/LowerContractionToGpu.cpp's
estimateRegistersPerThread() uses to prune kernel candidates analytically
before ever running the empirical runtime autotuner
(src/runtime/TutorialAutotuneRuntime.cpp).

Run: python3 src/benchmarks/tensor_parallel_placement_analysis.py
"""

from dataclasses import dataclass

BYTES_PER_ELEM = 4  # f32

# NVIDIA L4 public datasheet figures -- this project's own target GPU
# (docs/l4_evaluation_results.md: sm_89, driver 570.195.03, CUDA 12.8.1).
# These are vendor spec, not measured by this repo, and are used only as
# idealized peak-roof estimates, the same caveat the existing roofline
# doc makes about its own CPU numbers.
L4_FP32_PEAK_FLOPS = 30.3e12  # "shared-fp32" strategy, LowerContractionToGpu
L4_TF32_PEAK_FLOPS = 120e12  # "tensorcore-tf32" strategy, same pass

# PCIe Gen4 x16 -- already used as this project's own transfer-bandwidth
# assumption in docs/deep_dive_part3_qa6to10.md ("PCIe 4.0 (64 GB/s)").
# The L4 has no NVLink, so for this exact hardware this is not a
# simplifying assumption -- it is the only inter-GPU link that exists.
PCIE_GEN4_X16_BYTES_PER_SEC = 64e9


@dataclass
class FFNShape:
    name: str
    batch: int  # tokens
    d_model: int
    d_ffn: int


SHAPES = [
    # This repo's actual benchmarked shape: src/benchmarks/gpt2_ffn.mlir
    FFNShape("gpt2_ffn (this repo's benchmark)", batch=128, d_model=256, d_ffn=512),
    # Real GPT-2-small dimensions, for comparison against a toy shape.
    FFNShape("gpt2-small (real dimensions)", batch=128, d_model=768, d_ffn=3072),
]


def megatron_mlp_flops(shape: FFNShape) -> float:
    """Up-proj (d_model->d_ffn) + down-proj (d_ffn->d_model), forward only.
    Column-parallel up-proj needs no communication (each shard already
    holds exactly the d_ffn columns its row-parallel down-proj shard
    needs); row-parallel down-proj needs one all-reduce of the summed
    (batch, d_model) output. This is the standard Megatron-LM MLP split."""
    up = 2 * shape.batch * shape.d_model * shape.d_ffn
    down = 2 * shape.batch * shape.d_ffn * shape.d_model
    return up + down


def allreduce_bytes(shape: FFNShape) -> int:
    return shape.batch * shape.d_model * BYTES_PER_ELEM


def ring_allreduce_seconds(payload_bytes: int, n_gpus: int, link_bytes_per_sec: float) -> float:
    """Ring all-reduce cost model: each GPU moves 2*(N-1)/N * payload over
    its link (reduce-scatter, then all-gather). Assumes an idealized ring
    with no per-message latency floor and no link contention -- a lower
    bound on real communication time, same spirit as citing peak FLOPs
    rather than measured FLOPs for the compute side."""
    if n_gpus <= 1:
        return 0.0
    return 2 * (n_gpus - 1) / n_gpus * payload_bytes / link_bytes_per_sec


def compute_seconds(total_flops: float, n_gpus: int, peak_flops: float) -> float:
    return (total_flops / n_gpus) / peak_flops


def main():
    for shape in SHAPES:
        total_flops = megatron_mlp_flops(shape)
        payload = allreduce_bytes(shape)
        print(f"\n=== {shape.name}: batch={shape.batch} d_model={shape.d_model} d_ffn={shape.d_ffn} ===")
        print(f"total FLOPs (both matmuls, forward only): {total_flops / 1e6:.2f} MFLOPs")
        print(f"all-reduce payload per block: {payload} bytes ({payload / 1024:.1f} KiB)")
        print(f"{'N':>3} {'math':>6} {'compute/shard (us)':>20} {'all-reduce (us)':>18} "
              f"{'compute/comm ratio':>20} {'regime':>14}")
        for n_gpus in (1, 2, 4, 8):
            for math_mode, peak in (("fp32", L4_FP32_PEAK_FLOPS), ("tf32", L4_TF32_PEAK_FLOPS)):
                t_compute = compute_seconds(total_flops, n_gpus, peak)
                t_comm = ring_allreduce_seconds(payload, n_gpus, PCIE_GEN4_X16_BYTES_PER_SEC)
                ratio = t_compute / t_comm if t_comm else float("inf")
                regime = "compute-bound" if ratio > 1 else "comm-bound" if t_comm else "n/a (single GPU)"
                ratio_str = f"{ratio:.3f}" if t_comm else "n/a"
                print(f"{n_gpus:>3} {math_mode:>6} {t_compute * 1e6:>20.3f} {t_comm * 1e6:>18.3f} "
                      f"{ratio_str:>20} {regime:>14}")


if __name__ == "__main__":
    main()
