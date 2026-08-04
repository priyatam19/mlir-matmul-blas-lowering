# NVIDIA L4 Evaluation Results

## Run Configuration

The unified evaluation ran on 2026-08-04 on one idle NVIDIA L4 with 23,034
MiB, driver 570.195.03, CUDA 12.8.1, system cuDNN 9.8, and `sm_89`/PTX 8.0
code generation. Formal results are the median of three process-level trials,
each using 10 warmups and 50 timed samples. Diagnostics, Compute Sanitizer,
and tuning were run outside the formal timing interval.

The pod booted a generic CUDA image rather than the prepared image. The pinned
torch-mlir/LLVM stack and both Python environments were restored from the
prepared image and persistent workspace. The lowerings were built from
`8da8a9bed553e336d5bb6500fe5b4d222abebc4c`; runtime orchestration fixes ended
at `bc36a99f5671ecb2f974a4dedba4b2eb1cb24cd5`.

## Performance Gates

All formal gates passed:

| Gate | Result | Requirement |
|---|---:|---:|
| Long BMM custom speedup | 68.72x | at least 10x |
| ResNet 3x3 custom convolution speedup | 38.63x | at least 10x |
| BMM vendor/direct ratios | 1.036x-1.066x | at most 5x |
| Convolution vendor/direct ratios | 0.960x-1.033x | at most 5x |
| PR2 GEMM block-thread p50 | 0.135648 ms | 0.1206-0.1474 ms |

## Operator Timings

Canonical CUDA-event p50 values in milliseconds:

| BMM shape | Untiled | Block-thread | Vendor | Direct cuBLAS |
|---|---:|---:|---:|---:|
| BERT `12x128x64x128` | 1.036480 | 0.043968 | 0.017408 | 0.016800 |
| Long `12x512x64x512` | 21.816608 | 0.317472 | 0.045088 | 0.042304 |
| Value `32x128x64x64` | 1.391904 | 0.046624 | 0.017248 | 0.016608 |
| Irregular `7x129x65x127` | 0.645856 | 0.036128 | 0.018464 | 0.017792 |

| Convolution shape | Untiled | Block-thread | Vendor | Direct cuDNN |
|---|---:|---:|---:|---:|
| Small 3x3 | 0.087872 | 0.020160 | 0.030656 | 0.031040 |
| ResNet stem 7x7 | 7.789248 | 0.226976 | 0.061856 | 0.064448 |
| ResNet block 3x3 | 10.764224 | 0.278656 | 0.052544 | 0.052128 |
| Pointwise 1x1 | 1.899168 | 0.079936 | 0.117088 | 0.115328 |
| Irregular 3x5 | 0.181056 | 0.026688 | 0.033792 | 0.032704 |

The custom kernels close much of the launch-topology gap but remain slower
than vendor libraries on the compute-heavy BMM and convolution cases. This is
expected because they do not yet use workgroup memory, vector loads, or tensor
cores.

## Mixed Programs

| Program | Untiled | Block-thread | Vendor | PyTorch eager CUDA |
|---|---:|---:|---:|---:|
| Attention block | 3.936512 | 1.948736 | 1.897504 | 0.087232 |
| Residual convolution block | 3.129024 | 2.701440 | 2.293120 | 0.182592 |

The mixed programs retain many generic kernels, so replacing only BMM or
convolution leaves substantial launch and elementwise/reduction overhead. This
motivates fusion work after the next shared-memory operator optimization.

## Tuning

- BERT BMM favored `16x16` at 0.042688 ms; long BMM favored `8x32` at
  0.317312 ms.
- ResNet 3x3 favored 512 threads at 0.273856 ms; pointwise convolution favored
  64 threads at 0.077792 ms.
- The default `8x32` BMM and 256-thread convolution configurations remain
  reasonable cross-shape choices.

## Correctness and Diagnostics

- All full-output comparisons passed.
- All eight custom/vendor sanitizer targets reported zero errors.
- Custom isolated operators use one launch; vendor isolated operators use one
  cuBLAS or cuDNN call.
- Attention uses 14 generic/custom launches or 12 launches plus two cuBLAS
  calls. Residual convolution uses 9 launches or 7 plus two cuDNN calls.
- Nsight Compute could not access hardware counters (`ERR_NVGPUCTRPERM`), so
  profiling was recorded as non-blocking and skipped.

The run exposed and fixed three evaluation issues: sanitizer summaries may be
printed to stdout, `CUDA_HOME` must be exported so MLIR links `libdevice`, and
GPU telemetry CSVs must be excluded from benchmark parsing.

Compact machine-readable results are under
`runpod_results/l4_eval_2026-08-04`. The complete local archive is
`l4_eval_20260804T151913Z.tar.gz`, with SHA-256
`4e14dbf68df8e750c8f57422b293e2d1bc31f819eeb8abcaf448ddb126422107`.
