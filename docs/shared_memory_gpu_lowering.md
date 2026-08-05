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
| `autotuned` | strict FP32 candidate set | runtime-selected profile | exhaustive strict-FP32 candidates |

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
`scf.index_switch`. `tutorial_autotune_begin` selects exactly one candidate;
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
cache.

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
libraries. GPU performance numbers should only be added after this command is
run on the L4; offline compilation is not a performance result.
