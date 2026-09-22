# Shared-Memory and Tensor-Core GPU Lowering

## Scope

`--lower-contraction-to-gpu` lowers bufferized rank-2 GEMM, rank-3 batch
matmul, and NCHW/FCHW convolution into explicit GPU kernels. The original
`--tile-matmul-for-gpu`, `--tile-batch-matmul-for-gpu`, and
`--tile-conv2d-nchw-for-gpu` passes remain the portable block/thread baseline.

The optimized pass has three strategies:

| Strategy | Arithmetic | Staging | Intended comparison |
| --- | --- | --- | --- |
| `shared-fp32` | FP32 CUDA cores and FP32 accumulation | synchronous vector loads or two-stage `cp.async` | pedantic FP32 cuBLAS/cuDNN |
| `tensorcore-tf32` | TF32 multiply and FP32 accumulation | one or two workgroup stages | TF32 cuBLAS/cuDNN |
| `autotuned` | strict FP32 or TF32, selected by `AUTOTUNE_MATH_MODE` | runtime-selected profile | exhaustive candidates in the same math mode |

Unsupported element types, non-unit innermost strides, non-`sm_89` targets,
or unsupported linalg semantics remain available to the generic lowering
pipeline.

## Kernel Structure

GEMM and BMM use CTA M/N tiles, K tiles of 16 or 32, 128-256 threads, and
per-thread register microtiles. Cooperative 16-byte transfers stage A and B
in workgroup memory; edge vectors use masked scalar zero fill. Two-stage
profiles emit `nvgpu.device_async_copy`, group/wait operations, and alternating
workgroup buffers. BMM maps batch directly to `blockIdx.z`.

The TF32 path assigns `16x8` output tiles to warps. Lane-distributed A/B
fragments feed `nvgpu.mma.sync` with `mmaShape=[16,8,8]`; the pinned NVVM
pipeline lowers this to
`mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32` PTX.

Convolution is a direct implicit GEMM. It interprets filters as `[F,K]`, where
`K=C*KH*KW`, and output positions as `[K,P]`, where `P=OH*OW`. Input patches
are generated while filling workgroup memory, so no im2col tensor is
materialized. Stride, dilation, K tails, output-position tails, and input
bounds are handled in the kernel.

## Runtime Autotuning

`GPU_LOWERING=autotuned` emits eight eligible implementations and a host
`scf.index_switch`. Set `AUTOTUNE_MATH_MODE=shared-fp32` (the default) or
`AUTOTUNE_MATH_MODE=tensorcore-tf32` to choose the candidate family. The
benchmark harness also accepts `GPU_MATH_MODE=tf32` so an autotuned TF32 kernel
is validated and timed against TF32-enabled cuBLAS/cuDNN. The arithmetic mode
is part of the persistent cache key, so FP32 and TF32 winners cannot collide.
`tutorial_autotune_begin` selects exactly one candidate;
`tutorial_autotune_end` records its CUDA-event latency. Each candidate receives
two warmups and five samples, so an eight-candidate key converges after 56
invocations.

Winners are stored in
`~/.cache/mlir-tutorial/gpu-autotune-v1.csv`, keyed by GPU UUID, compute
capability, driver, compiler revision, candidate set, operator, and dynamic
dimensions. The runtime supports:

```text
TUTORIAL_AUTOTUNE_CACHE=/path/to/cache.csv
TUTORIAL_AUTOTUNE_READ_ONLY=1
TUTORIAL_AUTOTUNE_RESET=1
TUTORIAL_AUTOTUNE_DISABLE=1
TUTORIAL_AUTOTUNE_DEBUG=1
```

Cold convergence must be reported separately. Formal timings use a populated
cache. On a read-only cache miss, tuning-disabled run, or short-lived process,
candidate zero is an architecture-profile shared-memory or tensor-core default;
the portable block/thread implementation remains the final FP32 candidate.
For GEMM/BMM, offline `ptxas` inspection selects `64x128x16`, 256 threads as
the initial default: synchronous for strict FP32 and two-stage asynchronous for
TF32. The TF32 default was the tested candidate with zero spill loads/stores;
spill-heavy schedules remain measurable so runtime latency can reject them.

## Offline Verification

```bash
cmake --build build-ninja --target check-mlir-tutorial -j2

GPU_LOWERING=tensorcore-tf32 CONTRACTION_STAGES=2 \
CUDA_CHIP=sm_89 CUDA_PTX_FEATURE=+ptx80 \
MODEL_MLIR=src/benchmarks/gpu_gemm_1024.mlir \
bash src/sample/gpu/run_mlir_pipeline.sh

python3 src/benchmarks/extract_ptx.py \
  src/sample/gpu/sample_nvptx_isa.mlir /tmp/tutorial-ptx
ptxas --verbose --gpu-name sm_89 /tmp/tutorial-ptx/kernel_0.ptx \
  --output-file /tmp/tutorial-kernel.cubin
```

The lit suite checks workgroup allocations, vector and async transfers,
barriers, tail guards, FP32 register accumulation, BMM grid Z mapping,
implicit-GEMM convolution, runtime dispatch, and TF32 MMA IR. Offline PTX
checks require both `cp.async` and `mma.sync` and no remaining NVGPU ops.

## L4 Evaluation

The complete GPU run is prepared as one resumable command:

```bash
PROJ=/workspace/mlir_project \
OUTPUT_DIR=/workspace/mlir_project/runpod_results/shared_memory_gpu \
bash src/benchmarks/run_gpu_contraction_evaluation.sh
```

It runs three process trials, 10 warmups, and 50 CUDA-event samples for nine
GEMMs, four BMMs, five convolutions, attention, and a residual block. It also
runs profile sweeps, separates autotuner cold and steady-state measurements,
captures an Nsight Systems trace, extracts PTX, and records ptxas register,
spill, barrier, and shared-memory usage.

Strict FP32 uses `atol=rtol=1e-4` and relative L2 `<=1e-5`. TF32 uses
`atol=rtol=1e-2` and relative L2 `<=5e-3` against matching-mode vendor
libraries. TF32 CSV rows additionally report maximum absolute, maximum
relative, and relative-L2 drift from a separate untimed pedantic-FP32 vendor
reference.

## L4 Results (2026-09-22)

Run for the first time on a dedicated NVIDIA L4 (driver `580.126.20`, CUDA
12.8 toolkit, `sm_89`). Full methodology, two real bugs found and fixed on
hardware, one known bug found but deliberately not fixed, and every raw CSV
are in
[l4_shared_memory_gpu_evaluation_2026-09-22.md](l4_shared_memory_gpu_evaluation_2026-09-22.md).
Headline: this pass is a large, real improvement over the block-thread
baseline, closing much of the remaining gap to vendor libraries — but not a
finished result, since it's currently only validated on tile-aligned shapes.

GFLOP/s as a percentage of the matching vendor reference (higher is better),
p50 of 3 trials, tile-aligned shapes only (irregular/K-tail shapes excluded
this run — see the linked report for why):

| Shape family | block-thread (prior baseline) | shared-fp32 | autotuned | tensorcore-tf32 |
|---|---:|---:|---:|---:|
| GEMM (6 shapes, range) | 7.6%-39.1% | 24.8%-48.3% | 38.8%-50.8% | 13.1%-45.4% |
| BMM bert / long / value | 36.8% / 12.4% / 35.0% | 67.9% / 36.6% / 54.8% | - | - |
| Conv resnet-stem / resnet-block / pointwise | 21.8% / 11.8% / 93.8% | 42.9% / 18.3% / 221.4%\* | - | - |

\*Beats cuDNN outright on this shape, consistent with the block-thread
baseline's own note that dispatch overhead dominates cuDNN's advantage at
small/pointwise sizes.
