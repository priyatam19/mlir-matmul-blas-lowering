# GPU Block/Thread Matmul Benchmark

## Why This Pass Exists

The original GPU experiment reused `--tile-matmul-for-cache`, which creates
sequential `scf.for` loops around tensor tiles. For the
`(512x256)x(256x512)` benchmark with `16x16` output tiles, the host executes a
`32x32` loop nest. Every iteration launches a kernel containing 256 one-thread
blocks, for 1,024 launches per GEMM call.

The post-bufferization `--tile-matmul-for-gpu` pass instead emits:

```text
outer scf.parallel: output tiles -> CUDA blocks
inner scf.parallel: output elements -> CUDA threads
inner scf.for:      K reduction in a thread-local SSA accumulator
```

With `block-m=8 block-n=32` and `mapping-policy=innermost-first`, the N
dimension maps to CUDA X for contiguous memory access.

## Verified Lowering Structure

These properties were verified using the pinned torch-mlir/MLIR image without
requiring an NVIDIA driver:

| Mode | Host loops | Grid | Threads | Launches per call |
|---|---:|---:|---:|---:|
| `legacy` | `32x32` | `16x16x1` per iteration | `1x1x1` | 1,024 expected |
| `untiled` | none | `512x512x1` | `1x1x1` | 1 |
| `block-thread` | none | `16x64x1` | `32x8x1` | 1 |

The irregular `(513x257)x(257x509)` case lowers to grid `16x65x1` and threads
`32x8x1`. Fixed-size thread blocks are protected by row/column bounds checks.

All five benchmark shapes compile through GPU dialect, NVPTX for `sm_89`, LLVM
IR, object generation, and final linking against the MLIR CUDA runtime and
cuBLAS.

## RunPod L4 Procedure

Build or use the pinned image, mount the repository at
`/workspace/mlir_project`, and build `tutorial-opt`. Then run:

```bash
cd /workspace/mlir_project

GPU_LOWERING=legacy SHAPES=gemm_512 CHECK_LAUNCHES=1 \
  RUNS=10 bash src/benchmarks/run_gpu_gemm_benchmarks.sh

GPU_LOWERING=untiled SHAPES="gemm_512 gemm_1024 gemm_2048 gemm_mobilenet gemm_irregular" \
  CHECK_LAUNCHES=1 bash src/benchmarks/run_gpu_gemm_benchmarks.sh

GPU_LOWERING=block-thread SHAPES="gemm_512 gemm_1024 gemm_2048 gemm_mobilenet gemm_irregular" \
  BLOCK_M=8 BLOCK_N=32 CHECK_LAUNCHES=1 \
  bash src/benchmarks/run_gpu_gemm_benchmarks.sh

bash src/benchmarks/sweep_gpu_tiles.sh
```

`CHECK_LAUNCHES=1` uses `MLIR_CUDA_DEBUG=1` for one correctness invocation and
fails if `legacy` does not launch 1,024 kernels or the other modes do not launch
exactly one. It is separate from timed execution. Do not set
`CUDA_LAUNCH_BLOCKING=1` while collecting performance numbers.

The harness reports p10/p50/p90 CUDA-event time, p50 call-and-synchronize time,
GFLOP/s, and full-output error against pedantic-FP32 cuBLAS. Inputs and outputs
are preallocated managed memrefs and prefetched before timing.

## L4 Results

These measurements were collected on an NVIDIA L4 with CUDA 12.8.1. The
historical `56.524668 ms` result remains context only: the formal comparison
below reran all three modes with the same preallocated, CUDA-event harness.
Times are device-execution p50 in milliseconds.

| Shape | Legacy | Untiled | Block/thread | cuBLAS | Speedup vs untiled |
|---|---:|---:|---:|---:|---:|
| `512x256x512` | 55.069 | 6.725 | 0.134 | 0.027 | 50.1x |
| `1024x1024x1024` | not run | 107.419 | 1.759 | 0.172 | 61.1x |
| `2048x2048x2048` | not run | 858.592 | 16.499 | 1.143 | 52.0x |
| `3136x32x64` | not run | 0.508 | 0.029 | 0.011 | 17.7x |
| `513x257x509` | not run | 6.706 | 0.135 | 0.030 | 49.8x |

The `512x256x512` block/thread kernel is **410.6x faster** than the freshly
measured legacy p50, comfortably exceeding the 10x acceptance threshold. Its
throughput is `1,000.8 GFLOP/s`, versus `2.4 GFLOP/s` for legacy and
`20.0 GFLOP/s` for untiled. Launch diagnostics confirmed 1,024 legacy launches
and one launch for both untiled and block/thread modes.

All shapes passed a full-output comparison against pedantic-FP32 cuBLAS. The
largest observed absolute error was `2.29e-5` on the `2048x2048x2048` case;
the largest observed relative error was `4.42e-6`.

## Tile Sweep

The sweep used 20 timed samples per configuration:

| Tile | 512 p50 | 512 GFLOP/s | 1024 p50 | 1024 GFLOP/s |
|---|---:|---:|---:|---:|
| `8x32` | 0.133 | 1,008.7 | 1.762 | 1,218.8 |
| `16x16` | 0.133 | 1,012.1 | 1.918 | 1,119.4 |
| `16x32` | 0.134 | 1,000.8 | 1.850 | 1,161.1 |
| `32x8` | 0.134 | 1,003.4 | 1.816 | 1,182.6 |

`8x32` remains the default. Although `16x16` is 0.3% faster on the 512 case,
`8x32` is about 8% faster on the 1024 case and maps a complete warp along the
contiguous N dimension.

The runtime-weight PyTorch-derived sample also ran through `block-thread` on
the L4 and exactly matched its CPU output. Its matmul used one block with
`threads=(32,8,1)`; surrounding fill and elementwise kernels remained on the
generic lowering path.

## Current Limitation

This pass improves launch topology, contiguous thread mapping, and register
accumulation. It still reloads A and B from global memory for each output
thread. Shared-memory K tiling, vector operations, and tensor-core lowering are
deliberately left for the next optimization PR.
