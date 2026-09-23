# Shared-Memory/Tensor-Core GPU Lowering: L4 Evaluation (2026-09-22)

The first real-hardware run of `LowerContractionToGpuPass` (`agent/shared-memory-gpu-lowering`,
commit `50defad` at evaluation time). Every prior number for this pass was
offline (lit/FileCheck, `ptxas` register/spill inspection) — this is the run
`docs/shared_memory_gpu_lowering.md` said was still pending.

## Environment

- NVIDIA L4, 23,034 MiB, driver `580.126.20`, `sm_89`
- CUDA 12.8.93 toolkit (`nvcc`), CUDA 13.0 API reported by the driver
- torch-mlir/LLVM/MLIR built from source at the pinned commits
  (`TORCH_MLIR_COMMIT=c15565667004fe1f3726404afb3f3f862a945551`,
  `LLVM_COMMIT=068c6c5c0c8a0555036a2ff09a99f486548e6e8d` — same pins as
  `docker/Dockerfile.cuda`), since this RunPod instance had neither Docker
  nor the project's prepared image available as its base — see
  "Environment note" below.
- One run per configuration set; formal timing is the project's own
  3-trial/10-warmup/50-sample protocol, run by
  `src/benchmarks/run_gpu_contraction_evaluation.sh`

### Environment note: no prepared image on this pod

The pod provisioned for this run was a generic PyTorch/CUDA template
(30 GB writable disk, no Docker, no dedicated `/workspace` volume) rather
than the project's own image built from `docker/Dockerfile.cuda`, and
Docker-in-Docker is not available (`cap_sys_admin` is stripped from the
container's capability set, confirmed via `capsh`). torch-mlir + LLVM/MLIR
were built from source directly on the pod instead, restricted to the
target list `docker/Dockerfile.cuda` itself builds (`mlir-opt`,
`mlir-translate`, `llc`, `torch-mlir-opt`, `FileCheck`/`not`/`count`, the
CUDA runtime libraries), which fit in under 10 GB and completed well within
the 30 GB budget. `check-mlir-tutorial` (23/23 tests) passed against this
build before any GPU benchmarking started.

## Three real issues found running this for the first time

All three are genuine, hardware/first-run-surfaced issues, not artifacts of
the non-standard environment above — the environment note is included for
reproducibility, not as a caveat on the findings themselves.

### 1. Fixed: stale build artifact reuse in `run_gpu_gemm_benchmarks.sh`

`run_mlir_pipeline.sh` and `compile.sh` coordinate through a shared
`OUTPUT_DIR`/`SAMPLE_OBJECT`/`COMPILE_WORK_DIR` default (the script's own
directory) so consecutive shape builds don't collide.
`run_gpu_bmm_benchmarks.sh` and `run_gpu_conv_benchmarks.sh` already
isolate this correctly with a per-shape `artifact_dir`;
`run_gpu_gemm_benchmarks.sh` never did. It was latent until this evaluation
first ran shapes back-to-back with `OUTPUT_DIR` exported externally (exactly
as `docs/shared_memory_gpu_lowering.md`'s own example command does):
`gemm_512` built correctly, then `gemm_1024` silently linked against
`gemm_512`'s leftover `sample.o` and failed with `undefined reference to
_mlir_ciface_gpu_gemm_1024`. Fixed in commit `db857d2` the same way the
sibling scripts already do it.

### 2. Fixed: `CUDA_ERROR_MISALIGNED_ADDRESS` in two-stage async kernels

`nvgpu.device_async_copy` (`cp.async.128`, the `stages=2` pipeline's
transfer instruction) requires its shared-memory destination to be 16-byte
aligned and hardware-faults if it isn't — unlike `vector.load`/
`vector.store` (the `stages=1` path), which the compiler can silently split
into smaller aligned accesses when it can't prove alignment.
`sharedBColumns()` pads the B tile by +1 column whenever `blockK == 16`, to
rotate K rows across shared-memory banks. For every `blockN` this project
tests (64, 128 — both multiples of 4), `blockN + 1` is never itself a
multiple of 4, so the padded row stride is never 16-byte aligned. Harmless
for `stages==1`; a hard fault for `stages==2`.

First hit running the tuning sweep (`block=128x128x16 threads=256
stages=2`) and separately via the `autotuned` strategy's own `stages=2`
candidates (candidates 4-6 in the FP32 profile list). Confirmed by
inspecting the generated IR: `shared_b` was `memref<2x16x129xf32>` before
the fix (516-byte row stride, misaligned) and `memref<2x16x128xf32>` after
(512 bytes, aligned). Fixed in commit `50defad` by skipping the padding
specifically when `stages==2`, trading back some bank-conflict avoidance
for correctness on the async path. Verified directly on hardware (the exact
previously-crashing profile now runs and produces correct output) before
relaunching the full evaluation.

### 2b. Found, not fixed: `autotuned` gives wrong answers on non-tile-aligned K

`gemm_irregular` (513x257x509, K=257) and `gemm_ktail` (1024x1031x1024,
K=1031) both fail correctness under `autotuned` — `gemm_ktail` alone
mismatches 786,433 of 1,048,576 output elements (75%). Every shape with K
divisible by both 16 and 32 (512, 1024, 2048, mobilenet's K=32, tall, wide)
passes. The autotuner consistently selects the same winning candidate
(`{blockM=64, blockN=128, blockK=32, threads=256, stages=2}`, candidate
index 5 in `contractionProfiles()`) for every regular shape on this
GPU/driver/compiler combination — that candidate is correct for all of
them, and only breaks on non-tile-aligned K, pointing at a boundary/tail
guard bug specific to that profile (or to `blockK=32` K-tail handling in
general) rather than anything related to fix #2 above (this profile uses
`blockK=32`, which was never affected by the padding-alignment bug, and
doesn't crash — it silently computes the wrong answer). Reproduced twice.

**Not fixed in this session** — this needs its own bisection (which of the
tail guards in `emitAsyncTileLoads`/`emitCooperativeTileLoads` is wrong for
`blockK=32`, non-aligned K) rather than a quick patch, given the
alignment fix above already used the available time for one hardware-bug
investigation this run. `gemm_irregular`, `gemm_ktail`, `bmm_irregular`,
and `conv_irregular` were excluded from this evaluation's shape lists as a
result — `bmm_irregular`/`conv_irregular` were never actually tested this
run (excluded pre-emptively once the pattern was clear), so their status
under `autotuned` is unknown, not confirmed-fine.

### 3. Known limitation, not new: GEMM `vendor` mode has no real cuBLAS lowering

`--convert-batch-matmul-to-cublas` and `--convert-conv2d-nchw-to-cudnn`
exist; there is no rank-2-matmul-to-cuBLAS pass (the same gap
`docs/cpu_matmul_blas_ablation.md` found on the CPU side). `vendor` mode for
a plain GEMM shape therefore silently falls through to the generic,
untiled GPU lowering. This is slow but not otherwise broken for small
shapes (512, 1024, 2048 all completed, just slowly — see the raw
`gemm_vendor_trial*.csv` files, ~20 GFLOP/s flat regardless of shape,
matching the untiled kernel's known one-thread-per-output-element
characteristic); `gemm_4096` was excluded from this run because at that
scale the naive kernel takes long enough to make the run impractical
(estimated on the order of tens of minutes for a single build+time cycle,
extrapolating from the 2048/1024/512 progression). **Do not read the
`vendor` rows in `gemm_*.csv` as cuBLAS performance** — they are the
generic fallback path timed under a misleading label. BMM and conv vendor
rows are real cuBLAS/cuDNN.

## Results

GFLOP/s, p50 of 3 trials (or single-trial where noted), tile-aligned shapes
only. Raw CSVs for every family/mode/trial are in this directory's sibling
`runpod_results/l4_shared_memory_gpu_2026-09-22/` archive.

### GEMM (6 shapes: 512, 1024, 2048, mobilenet, tall, wide — `gemm_irregular`/`gemm_ktail`/`gemm_4096` excluded)

| Shape | block-thread | shared-fp32 | autotuned | tensorcore-tf32 | cuBLAS pedantic-fp32 | cuBLAS tf32 |
|---|---:|---:|---:|---:|---:|---:|
| 512 | 1033 (18.6%) | 1933 (34.7%) | 2832 (50.8%) | 2197 (20.0%) | 5563 | 11009 |
| 1024 | 964 (8.0%) | 3870 (29.9%) | 5159 (40.6%) | 4433 (18.6%) | 12948 | 23891 |
| 2048 | 970 (7.6%) | 3640 (26.9%) | 5428 (41.9%) | 5008 (13.1%) | 13554 | 38359 |
| mobilenet (3136x32x64) | 532 (39.1%) | 746 (48.3%) | 633 (47.2%) | 748 (45.4%) | 1544 | 1645 |
| tall (4096x1024x256) | 977 (8.8%) | 3203 (26.9%) | 5042 (42.1%) | 4456 (16.0%) | 11909 | 27858 |
| wide (256x1024x4096) | 981 (8.2%) | 3197 (24.8%) | 5025 (38.8%) | 4449 (14.8%) | 12911 | 29999 |

Percentages are share of the matching vendor reference (pedantic-FP32 for
block-thread/shared-fp32/autotuned-fp32, TF32 for tensorcore-tf32).
`autotuned` beats plain `shared-fp32` everywhere (as it should — it's
selecting the best of several `shared-fp32`-family candidates), and both
beat `block-thread` on every GEMM shape: **shared-fp32 is 1.4x-4.0x
block-thread's GFLOP/s** across these six shapes (1.4x mobilenet, 1.9x 512,
3.3x tall/wide, 3.8x 2048, 4.0x 1024); **autotuned is 1.2x-5.6x** (its
smallest gain, 1.2x, is mobilenet, where it is actually slower than plain
`shared-fp32`; its largest, 5.6x, is 2048). The large gains are on the large
shapes; the small mobilenet shape gains little.
`tensorcore-tf32` closes less of its (much larger) gap to `cublas-tf32` than
the FP32 strategies do — cuBLAS's TF32 path is dramatically faster than its
FP32 path (2x-2.8x, real tensor-core throughput), while this project's
`tensorcore-tf32` kernel is only modestly faster than its own `shared-fp32`
(1.1x-1.4x), so the vendor-relative percentage drops even though the
absolute GFLOP/s is higher.

### BMM (3 shapes: bert, long, value — `bmm_irregular` excluded)

| Shape | block-thread | shared-fp32 | tensorcore-tf32 | vendor (real cuBLAS) | cuBLAS pedantic-fp32 | cuBLAS tf32 |
|---|---:|---:|---:|---:|---:|---:|
| bert (12x128x64x128) | 631 (36.8%) | 1163 (67.9%) | 1273 (41.9%) | 1652 | 1713 | 3036 |
| long (12x512x64x512) | 1222 (12.4%) | 3603 (36.6%) | 3403 (24.2%) | 9218 | 9838 | 14059 |
| value (32x128x64x64) | 795 (35.0%) | 1245 (54.8%) | 1229 (33.9%) | 2175 | 2275 | 3628 |

Unlike GEMM, BMM's `vendor` mode is real (`--convert-batch-matmul-to-cublas`
exists), and its throughput here (1652-9218 GFLOP/s) tracks the pedantic-fp32
reference closely (within a few percent), confirming the vendor pass itself
is working correctly.

### Convolution (4 shapes: small, resnet-stem, resnet-block, pointwise — `conv_irregular` excluded)

| Shape | block-thread | shared-fp32 | vendor (real cuDNN) | cuDNN pedantic-fp32 |
|---|---:|---:|---:|---:|
| small | 116 (117.7%) | 104 (105.1%) | 96 (97.4%) | 99 |
| resnet-stem | 1028 (21.8%) | 2029 (42.9%) | 4924 (104.0%) | 4735 |
| resnet-block | 819 (11.8%) | 1276 (18.3%) | 6921 (98.4%) | 7035 |
| pointwise | 672 (93.8%) | 1585 (221.4%) | 700 (97.7%) | 716 |

`shared-fp32` improves `resnet-stem` 2.0x and `resnet-block` 1.6x over
`block-thread` (vendor-relative share 21.8% -> 42.9% and 11.8% -> 18.3%),
and turns `pointwise` from slightly behind cuDNN into beating it 2.2x
(2.4x over block-thread). It is **not** uniformly better: `small` is ~11%
slower under `shared-fp32` than under `block-thread` (104 vs. 116 GFLOP/s),
plausibly because at that size the kernel is launch/latency-bound and the
shared-memory staging is pure overhead (not investigated further). `small` and `pointwise` beating
or matching cuDNN in every mode (including the real vendor path) is
consistent with `docs/shared_memory_gpu_lowering.md`'s and the earlier
block-thread evaluation's own observation: cuDNN's dispatch/algorithm-search
overhead isn't amortized at small/pointwise sizes.

### Mixed programs (attention block, residual conv block)

| Program | shared-fp32 | tensorcore-tf32 | autotuned |
|---|---:|---:|---:|
| attention_block (ms) | 1.528 | 1.316 | 1.578 |
| residual_conv_block (ms) | 1.736 | 2.182 | 2.131 |

`autotuned` is not uniformly the winner for a *mixed* program the way it is
for isolated operators — attention's `autotuned` (1.578 ms) is slower than
both `shared-fp32` (1.528 ms) and `tensorcore-tf32` (1.316 ms), and
residual's `shared-fp32` (1.736 ms) beats both TF32 variants. These programs
mix the mapped GEMM/conv with several generic (unmapped) kernels — matching
the project's own prior Nsight finding that the mapped operator is a small
fraction of total program time, so per-operator autotuning gains can be
swamped by everything else in the program, and TF32's benefit (real for
isolated large GEMM) doesn't necessarily transfer once the operator itself
is a small piece of the whole.

### Tuning sweep and autotune validation

The exhaustive profile sweep (7 FP32 x 2 shapes + 8 TF32 x 2 shapes = 30
configurations x `gemm_512`/`gemm_1024`) completed cleanly after the
alignment fix, including the exact previously-crashing
`128x128x16/stages=2` configuration for both `shared-fp32` and
`tensorcore-tf32`. `analyze_contraction_results.py`'s autotune validation
(comparing the autotuner's chosen winner against the sweep's actual best)
**passed** for both math modes.

## What this run does and doesn't establish

**Established:** the shared-memory/tensor-core kernels are a large, real
improvement over the block-thread baseline on the medium/large
tile-aligned shapes (up to 4.0x for `shared-fp32` and 5.6x for `autotuned`
on GEMM, 1.6x-3.0x on BMM, 1.6x-2.4x on the larger convs),
correctness-verified against pedantic-FP32/TF32 vendor references, with the
autotuner selecting the sweep's best candidate. The gain is small on the
smallest shapes (GEMM mobilenet 1.4x) and negative on `conv_small` (-11%).
Comparisons are against *pedantic-FP32* (no TF32 promotion) vendor
libraries, and only the FP32 strategies are near the vendor number:
`tensorcore-tf32` reaches only 13%-45% of cuBLAS-TF32.

**Not yet established:** correctness on non-tile-aligned shapes under
`autotuned` (known-broken, see #2b), correctness of `bmm_irregular`/
`conv_irregular` under any mode this run (untested), Compute Sanitizer
coverage (not run this session — time was spent on the two bugs above
instead), Nsight Systems profiling (not run), and `gemm_4096`'s real
`vendor`-mode comparison (impossible until GEMM gets a real
matmul-to-cuBLAS pass — see #3).
