# GPU Convolution Benchmarks

## Lowering Modes

PR 4 extends the post-bufferization GPU pipeline to f32
`linalg.conv_2d_nchw_fchw` operations with standard indexing maps:

| Mode | Convolution behavior |
|---|---|
| `untiled` | Generic linalg parallel-loop lowering |
| `block-thread` | Linearized N/F/OH/OW mapped into fixed 256-thread blocks |
| `vendor` | Cached cuDNN forward-convolution runtime call |

The custom pass accepts static, dynamic, identity, and strided rank-4
memrefs. It decodes each output coordinate, guards the final partial block,
accumulates IC/KH/KW in SSA, and preserves `C += convolution`. Positive
constant strides and dilations are supported. Grouped, depthwise, and NHWC
convolutions remain outside this pass.

The vendor pass uses a canonical dynamic-strided memref ABI. Its runtime
honors memref offsets and input/output tensor strides, requires a contiguous
FCHW filter, preserves the initial output with cuDNN beta=1, and caches the
handle, descriptors, selected algorithm, and workspace by shape. The default
workspace limit is 256 MiB and can be changed with
`CUDNN_WORKSPACE_LIMIT_MB`.

## Pinned Environment

- CUDA image: `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04`
- Resolved image digest:
  `sha256:ad6d59a3bbf3e82c1c849c9ac09cfc2a3e0bbb8655042fd899be6681b3fe2a85`
- cuDNN: `9.8.0.87-1` for CUDA 12
- Compiler PyTorch: `2.13.0+cpu`; torchvision: `0.28.0+cpu`
- Eager baseline PyTorch: `2.8.0+cu128`; NumPy: `2.1.2`
- torch-mlir: `c15565667004fe1f3726404afb3f3f862a945551`
- LLVM: `068c6c5c0c8a0555036a2ff09a99f486548e6e8d`
- L4 code generation: `sm_89`, PTX 8.0

## Benchmark Programs

| Name | Logical feature map | Filter | Stride/dilation | Convolution input/output |
|---|---|---|---|---|
| `conv_small` | N4 C1 28x28 | F32 3x3 | 1x1 / 1x1 | pre-padded 30x30 -> 28x28 |
| `conv_resnet_stem` | N1 C3 224x224 | F64 7x7 | 2x2 / 1x1 | pre-padded 230x230 -> 112x112 |
| `conv_resnet_block` | N1 C64 56x56 | F64 3x3 | 1x1 / 1x1 | pre-padded 58x58 -> 56x56 |
| `conv_pointwise` | N1 C32 112x112 | F64 1x1 | 1x1 / 1x1 | 112x112 -> 112x112 |
| `conv_irregular` | N2 C7 35x37 | F13 3x5 | 2x1 / 2x1 | 35x37 -> 16x33 |

The isolated fixtures receive already padded input where a framework model
normally pads. This keeps the operator timing focused on convolution. The
PyTorch-derived residual block retains `tensor.pad` as a separate operation,
matching torch-mlir's actual lowering.

Each operator benchmark uses managed preallocated rank-4 memrefs, prefetches
them to the GPU, resets output outside the timed interval, and compares every
element against FP32 cuDNN. It reports CUDA-event and wall-clock p10/p50/p90,
GFLOP/s, and maximum absolute/relative error. The residual block uses runtime
weights and contains two pads, two convolutions, bias initialization, ReLU,
residual addition, and final ReLU. The residual runner generates deterministic
inputs shared with PyTorch and compares every compiled output element with a
PyTorch-produced reference tensor.

## L4 Procedure

Build and launch the pinned CUDA image on an NVIDIA L4, then run:

```bash
cd /workspace/mlir_project

for mode in untiled block-thread vendor; do
  GPU_LOWERING="${mode}" CHECK_LAUNCHES=1 \
    RESULTS_FILE="conv_${mode}.csv" \
    bash src/benchmarks/run_gpu_conv_benchmarks.sh
done

for mode in untiled block-thread vendor; do
  GPU_LOWERING="${mode}" CHECK_LAUNCHES=1 \
    bash src/benchmarks/run_gpu_residual_conv_benchmark.sh
done

/opt/pytorch-cuda/bin/python \
  src/benchmarks/pytorch_residual_conv_bench.py
```

Run diagnostics separately from timing. `MLIR_CUDA_DEBUG=1` confirms one
custom launch for each isolated convolution. In vendor mode,
`TUTORIAL_GPU_RUNTIME_DEBUG=1` reports one cuDNN call per convolution along
with its algorithm and workspace. The compiler environment generates the
PyTorch reference; the separate `/opt/pytorch-cuda` environment is used only
for eager-CUDA timing. Do not enable `CUDA_LAUNCH_BLOCKING=1` during timed
samples.

## Offline Verification

- All five operator shapes generate `sm_89` NVPTX in all three modes.
- Generic and custom operator paths each contain one outlined GPU launch.
- Vendor operator paths replace convolution with one runtime call.
- The mixed residual program generates 9 launches in generic/custom modes.
- Vendor residual lowering retains 7 generic launches and emits 2 cuDNN calls.
- The C++ harnesses and cuDNN runtime compile with warnings as errors.
- `check-mlir-tutorial` passes 14/14 tests.

The 2026-08-04 L4 run passed full-output validation and Compute Sanitizer. The
ResNet 3x3 convolution improved from 10.764224 ms untiled to 0.278656 ms
block-thread, a 38.63x speedup. Its vendor path measured 0.052544 ms versus
0.052128 ms for direct cuDNN. See [the unified L4 results](l4_evaluation_results.md)
for every shape, mixed-residual measurements, tuning, and artifacts.
