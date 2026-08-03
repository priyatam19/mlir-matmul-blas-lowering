# Benchmark Results

This project extends the MLIR tutorial pipeline with a custom matmul tiling pass
and evaluates two questions:

1. Does the transformed MLIR still execute correctly on CPU and CUDA paths?
2. When the workload is large enough, where does the GPU path become valuable,
   and where does the custom tiling pass help?

## Environment

RunPod PyTorch 2.8.0 pod:

- GPU: NVIDIA L4, 24 GB VRAM
- CUDA target used by MLIR: `CUDA_CHIP=sm_89 CUDA_PTX_FEATURE=+ptx80`
- CPU backend: MLIR lowered to LLVM and linked against OpenBLAS
- GPU backend: MLIR GPU/NVPTX lowering linked with `libmlir_cuda_runtime`

CPU-only tiling checks were also run inside the local Docker image with
OpenBLAS configured single-threaded.

## PyTorch-Derived Runtime-Weight Sample

The original GPU lowering path failed when dense constants from the
PyTorch-derived model were materialized as host LLVM globals and then referenced
from CUDA kernels. The fixed sample path externalizes weights, bias, and
input-shift constants as runtime memrefs and allocates them with CUDA managed
memory in the GPU harness.

| Backend | Pipeline | Avg Inference Time |
|---|---|---:|
| CPU | tiled MLIR + OpenBLAS | 0.000001843 sec |
| GPU | tiled MLIR + MLIR CUDA runtime on L4 | 0.000174743 sec |

Both paths produced the same output:

```text
0.38436 0.00000 0.24253 0.22031 0.00000
0.92511 0.00000 0.63827 0.48678 0.00000
0.93357 0.00000 0.31702 0.44530 0.00000
```

Interpretation: this tiny model is expected to be faster on CPU because GPU
launch overhead dominates the compute. The important milestone is correctness:
PyTorch-derived MLIR can go through the custom pass, lower to LLVM/NVPTX, link,
launch, and match the CPU result.

## Large GEMM GPU Check

Workload:

```text
(512x256) x (256x512) -> (512x512)
```

| Backend | Avg Time | Notes |
|---|---:|---|
| CPU MLIR + OpenBLAS | ~0.00216 sec | Native CPU executable |
| Generic MLIR GPU lowering | 0.056524668 sec | Correct, but not an optimized GEMM kernel |
| cuBLAS GPU baseline | 0.000021057 sec | Same input/checksum, tuned vendor kernel |

Checksum sample for the GPU paths:

```text
0.618624 0.710140 0.645500
```

Interpretation: the L4 GPU has a large performance advantage for real GEMM
workloads, but the generic MLIR GPU loop lowering is not competitive with
cuBLAS. The next optimization step is to lower recognized matmul operations to
cuBLAS, or implement a proper tiled/threaded CUDA kernel lowering instead of
launching generic loop kernels.

## Tiling Benefit On CPU/OpenBLAS

The custom pass was also evaluated on a MobileNet-shaped pointwise convolution
expressed as GEMM:

```text
(3136x32) x (32x64) -> (3136x64)
```

| Configuration | p50 | p10 | p90 |
|---|---:|---:|---:|
| Non-tiled MLIR + OpenBLAS | 1,208,243 ns | 1,167,013 ns | 1,490,181 ns |
| Tiled MLIR + OpenBLAS | 662,446 ns | 628,676 ns | 995,044 ns |

Observed p50 speedup:

```text
1.82x
```

Interpretation: this is the best current evidence that the custom tiling pass is
useful. The pass splits a larger matmul into cache-sized subproblems before the
BLAS lowering. For this shape, the full input/output working set is much larger
than the per-tile working set, so tiling improves locality enough to overcome
the overhead of issuing multiple smaller BLAS calls.

## Current Takeaways

- The CPU/OpenBLAS path is functional and benefits from tiling on selected
  real model shapes.
- The CUDA path now links and launches correctly for runtime-weight MLIR models.
- GPU acceleration is clearly valuable for large GEMM when using a tuned kernel
  such as cuBLAS.
- Generic MLIR GPU lowering is currently a correctness baseline, not a
  performance baseline.
- The most logical next pass is a matmul-to-cuBLAS lowering path, followed by a
  CUDA-specific tiled matmul lowering if the project wants to demonstrate a
  custom GPU kernel instead of a library call.

## GPU Block/Thread Follow-Up

The follow-up implementation adds a post-bufferization
`--tile-matmul-for-gpu` pass. It maps output tiles to CUDA blocks, output
elements to CUDA threads, and carries the K reduction in an SSA accumulator.
For the `512x256x512` benchmark, offline lowering changes the launch topology
from 1,024 expected runtime launches to one launch with grid `16x64x1` and
threads `32x8x1`.

The same-harness RunPod L4 comparison measured `55.069 ms` for legacy,
`6.725 ms` for untiled generic lowering, and `0.134 ms` for block/thread
lowering. The custom kernel is therefore `410.6x` faster than legacy and
`50.1x` faster than untiled for this shape. Runtime diagnostics confirmed that
legacy performs 1,024 launches while the custom path performs one.

All five benchmark shapes passed full-output comparison against pedantic-FP32
cuBLAS, including the irregular `513x257x509` case. The block/thread kernel
reached roughly `1.0-1.2 TFLOP/s` on the regular GEMMs, while cuBLAS remained
substantially faster. See
[gpu_block_thread_benchmarks.md](gpu_block_thread_benchmarks.md) for the full
table, tile sweep, and launch-validation procedure.
