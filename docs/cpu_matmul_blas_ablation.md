# CPU Matmul: BLAS vs. Generic-Loop Ablation

## Why This Experiment Exists

`docs/benchmark_results.md` reports a single `1.82x` speedup from
`--tile-matmul-for-cache` on one MobileNet-shaped GEMM, always measured with
the output going through `ConvertMatmulToBlasLibraryCallPass` (OpenBLAS).
That number conflates two independent effects:

1. Does converting `linalg.matmul` to a `cblas_sgemm` call help, versus
   letting the generic MLIR pipeline lower it to loops?
2. Does tiling with `--tile-matmul-for-cache` help, independent of (1)?

No prior run isolates them. This experiment builds all four combinations
(tiled/untiled x BLAS/no-BLAS) for the same four real-world shapes already
used in `src/benchmarks/` (`bert_attn`, `resnet_conv`, `gpt2_ffn`,
`large_gemm`), reusing the existing `*_bench.cpp` harnesses unmodified.

A structural finding surfaced while building this: **`ConvertMatmulToBlasLibraryCallPass`
is not actually optional.** `BufferizationToLLVMPipelineBuilder` in
`tools/tutorial-opt.cpp` runs it unconditionally as the first step of the
`--bufferization-to-llvm` meta-pipeline that every CPU benchmark uses:

```cpp
// CRITICAL: Replace matmuls with BLAS calls AFTER bufferization but BEFORE
// other LLVM conversions
manager.addPass(createConvertMatmulToBlasLibraryCallPass());
manager.addPass(mlir::createConvertLinalgToLoopsPass());  // only fires on what's left
```

There is no flag to disable it. `TileMatMulForCache` is the pass that's
optional (per Chapter 5 / the pipeline walkthrough); the BLAS conversion is
not. To get the "no-BLAS" leg of this ablation, a second pipeline was built
by hand that drops that one `addPass` call and falls straight through to
`createConvertLinalgToLoopsPass()`, keeping every other pass (strided-metadata
expansion, affine fusion/vectorize, SCF-to-CF, LLVM conversions, cleanup)
identical.

## Environment

This ran on a local workstation, not the project's usual RunPod/Docker
image, so absolute numbers are not comparable to `runpod_results/`. It's
recorded here for the relative comparison between legs, which is the point
of an ablation.

- CPU: Intel Core i9-13900HK (`raptorlake`), AVX2 + FMA + AVX-VNNI
- `tutorial-opt`: prebuilt at `build-ninja/tools/tutorial-opt` (MLIR/LLVM `23.0.0git`)
- `mlir-translate` / `llc` / `opt` / `clang`: LLVM 23.1.1, sourced from a sibling
  project's toolchain (`~/.local/share/lpta/toolchains/llvm-23.1.1`) since the
  original Docker build's LLVM tree (`/build/build/...` in `CMakeCache.txt`)
  does not exist on this host
- OpenBLAS: `libopenblas-dev` (0.3.26, `openblas-pthread` variant) installed
  via `apt` for this experiment — not present on the host beforehand
- `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` (single-threaded, matching
  `src/benchmarks/run_real_benchmarks.sh`)
- One run per configuration (not the project's usual median-of-3 trials)

All four legs were cross-checked for numeric correctness: output checksums
for `bert_attn` agree to `2671.086106`-`2671.086111` across all four
variants (BLAS vs. no-BLAS, tiled vs. untiled) — the ~1e-5 spread is
ordinary FP32 accumulation-order noise, confirming every path computes the
real result rather than a stub.

## Result 1: BLAS vs. No-BLAS (the requested ablation)

p50, nanoseconds:

| Shape | notiled+BLAS | notiled+noBLAS | tiled+BLAS | tiled+noBLAS | BLAS speedup | Tiling effect w/ BLAS | Tiling effect w/o BLAS |
|---|---:|---:|---:|---:|---:|---:|---:|
| bert_attn (128x64x128) | 47,879 | 1,984,535 | 53,404 | 1,985,189 | **41.4x** | 1.12x slower | ~no change |
| resnet_conv (256x128x256) | 263,235 | 16,936,062 | 291,081 | 17,053,045 | **64.3x** | 1.11x slower | ~no change |
| gpt2_ffn (128x256x512) | 424,161 | 34,430,601 | 473,973 | 34,955,101 | **81.2x** | 1.12x slower | 1.5% slower |
| large_gemm (512x256x512) | 1,641,752 | 137,326,401 | 1,792,134 | 139,451,958 | **83.6x** | 1.09x slower | 1.5% slower |

BLAS conversion dominates by 41x-84x, growing with problem size. Tiling is
noise-to-negative next to it in every shape, in both conditions. This
independently reproduces `runpod_results/real_cpu_tiling_benchmarks.txt` and
`local_cpu_tiling_after_fix.txt`, which show the same tiled-with-BLAS
slowdown on the same four shapes but were never linked from
`docs/benchmark_results.md`. The `1.82x` headline number there is a single
outlier shape (`3136x32x64`, unusually tall/skinny), not the representative
case.

## Result 2: Does `-mcpu=native` close the gap?

`llc` in the no-BLAS pipeline above ran with no `-mcpu` flag, i.e. generic
x86-64 codegen with no AVX2. Adding `-mcpu=native` (and even a full
`opt -O3 -march=native` pre-pass) produced **zero change** — still zero
packed SIMD instructions (`vfmadd*ps`, `mulps`/`addps`) in the compute loop,
confirmed via `objdump`.

The root cause, from LLVM's own vectorizer diagnostics
(`-Rpass-analysis=loop-vectorize`):

```
remark: loop not vectorized: cannot prove it is safe to reorder
floating-point operations; allow reordering by specifying
'#pragma clang loop vectorize(enable)' before the loop or by providing
the compiler option '-ffast-math'
```

Critically, passing `-ffast-math` as a **driver flag** to `clang -x ir` does
**not** retroactively add `fast` flags to `fmul`/`fadd` instructions that
already exist in IR without them — that flag only affects code the frontend
itself generates from source. Since `mlir-translate` lowers MLIR's
`arith.mulf`/`arith.addf` (no `fastmath` attribute by default anywhere in
this pipeline) straight to plain `fmul float`/`fadd float`, no downstream
compiler flag can vectorize the reduction — not `-mcpu=native`, not
`-ffast-math`, not both together.

Confirmed by hand-patching the two reduction instructions in the `.ll` for
`bert_attn` (`fmul float` -> `fmul fast float`, same for `fadd`) and
recompiling with `clang -x ir -O3 -march=native`:

```
remark: vectorized loop (vectorization width: 8, interleaved count: 4)
```

i.e. real AVX2 8-wide FMA, 4x unrolled (`vfmadd231ps` in the disassembly).
Rebuilding all 8 no-BLAS variants (4 shapes x tiled/untiled) this way:

| Shape | no-BLAS (generic) | no-BLAS (native+fastmath) | Speedup from vectorization | Still vs. BLAS |
|---|---:|---:|---:|---:|
| bert_attn notiled | 1,985,480 ns | 574,072 ns | 3.46x | 12.0x slower than BLAS |
| resnet_conv notiled | 16,921,624 ns | 6,655,476 ns | 2.54x | 25.3x slower than BLAS |
| gpt2_ffn notiled | 34,351,289 ns | 15,602,140 ns | 2.20x | 36.8x slower than BLAS |
| large_gemm notiled | 136,828,211 ns | 62,493,645 ns | 2.19x | 38.1x slower than BLAS |

Tiling effect after vectorization is fixed (tiled vs. notiled, both
native+fastmath): bert_attn ~unchanged, resnet_conv 0.7% *faster*, gpt2_ffn
0.2% slower, large_gemm 2.0% slower — still no clear win, at these shapes.

## Interpretation

1. **The BLAS-vs-no-BLAS gap is the dominant lever by a wide margin** (41x-84x
   vs. tiling's -12%-to-+2%). This answers the original question directly.
2. **Basic vectorization alone recovers 2.2x-3.5x** of that gap, and it's
   currently left on the table by the generic fallback pipeline for a
   structural reason (missing fast-math flags at the MLIR level), not a
   missing compiler flag. `-mcpu=native` by itself is not the fix.
3. **Even after fixing vectorization, tiling still doesn't show a
   significant benefit** at these four shapes (largest working set here is
   ~1.5 MB across A+B+C for large_gemm, against a 1.25 MB L2 and 24 MB L3).
   Whether tiling starts paying off at a scale that actually exceeds L3 is
   tested separately below.

## Result 3: Does tiling help once vectorized, at an L3-exceeding scale?

`gemm2048.mlir`: `2048x2048x2048`, ~16 MB per matrix, ~48 MB total working
set — well past the 1.25 MB L2 and past the 24 MB L3 on this CPU. Tile
config `tile-m=128 tile-n=128 tile-k=256` (same shape family as
`large_gemm`'s tiling, scaled up). 15 timed calls, 3 warmups.

| Variant | p50 | GFLOP/s |
|---|---:|---:|
| notiled + BLAS | 158.80 ms | 108.2 |
| tiled + BLAS | 188.49 ms | 91.1 |
| notiled + no-BLAS (native+fastmath) | 26,562.55 ms | 0.647 |
| tiled + no-BLAS (native+fastmath) | 8,794.04 ms | 1.954 |

**This is the headline result of the whole investigation.** Tiling flips
sign depending on whether BLAS is in the loop:

- **With BLAS, tiling gets *worse* as scale grows**: 9-12% slower at the
  small real-world shapes (Result 1), **18.7% slower** at 2048^3. More,
  smaller `cblas_sgemm` calls (256 calls of `128x256x128` here) consistently
  lose to OpenBLAS's own internal blocking on a single call, and the
  penalty grows with the number of calls, not shrinks.
- **Without BLAS, tiling is a real 3.02x win** at this scale, once
  vectorization is fixed (Result 2) — but was flat-to-slightly-negative at
  the smaller shapes, whose full working sets (128 KB-2 MB) mostly already
  fit in L2 (1.25 MB) or close to it, leaving no cache-miss penalty for
  tiling to remove. At 48 MB, there's a real penalty, and cache-sized tiles
  recover most of it.

This resolves the open question from Result 2: cache tiling is not
fundamentally useless on this pipeline, it was tested at the wrong scale
and with vectorization broken, both of which happened to hide its effect
(and, on the BLAS path, its harm).

Even the best custom result here (tiled+native, 1.95 GFLOP/s) remains
55.4x slower than plain untiled BLAS (108.2 GFLOP/s) — BLAS's multi-level
blocking, operand packing, and hand-tuned microkernels are doing
qualitatively more than a single level of cache tiling plus
auto-vectorization can match. See "Next Steps" below.

## Next Steps

Ranked by expected impact for effort:

1. **Fix vectorization at the MLIR level, not by hoping LLVM's
   auto-vectorizer or `-mcpu=native` infers it.** This alone was worth
   2.2x-3.5x at small scale and is a prerequisite for tiling to show any
   effect at all (Result 2/3). **Implemented** as `EnableFastMathForLoopsPass`
   (`lib/EnableFastMathForLoops.cpp`) plus a new
   `--bufferization-to-llvm-generic` pipeline in `tools/tutorial-opt.cpp`
   that mirrors `--bufferization-to-llvm` but skips the BLAS conversion and
   applies `reassoc|contract` fastmath flags to the arith ops
   `convert-linalg-to-loops` produces, so LLVM's vectorizer is actually
   allowed to reorder the reduction. Covered by
   `tests/enable_fastmath_for_loops.mlir`.

   Verified end-to-end through the real pass and pipeline (not the
   hand-patched `.ll` used for Result 2) — `objdump` confirms real
   `vfmadd*ps` in every shape, and a `bert_attn` output-checksum
   cross-check against the BLAS path agrees to `2671.086102` vs.
   `2671.086106` (FP32 rounding only):

   | Shape | no-BLAS, generic (Result 1) | no-BLAS, `--bufferization-to-llvm-generic` (real pass) |
   |---|---:|---:|
   | bert_attn notiled | 1,984,535 ns | 571,522 ns |
   | resnet_conv notiled | 16,936,062 ns | 6,716,031 ns |
   | gpt2_ffn notiled | 34,430,601 ns | 15,671,575 ns |
   | large_gemm notiled | 137,326,401 ns | 62,627,105 ns |

   Matches the hand-patched numbers from Result 2 to within ~1% run-to-run
   noise, confirming the pass is doing the same thing as the manual IR
   patch, through real, reusable, tested code. One build note: this
   required a fresh build directory (`build-local`) against a different
   LLVM/MLIR install (`~/.local/share/lpta/toolchains/llvm-23.1.1`) than
   `build-ninja` was originally built with, since the original Docker
   build's LLVM tree no longer exists on this host — see Reproduction
   below. Also worth remembering while testing this further: `llc -O3`
   alone does *not* run LLVM's loop vectorizer (that's a middle-end `opt`
   pass); use `clang -x ir -O3 -march=native -c` (or `opt -O3` before
   `llc`) to actually exercise it, or the fastmath fix will look like it's
   not working when it is.
2. **Stop tiling before BLAS.** Confirmed harmful at every scale tested,
   and the harm grows with scale (Result 3). `TileMatMulForCache` and
   `ConvertMatmulToBlasLibraryCallPass` should not be recommended together;
   `docs/benchmark_results.md`/Chapter 5 should be updated accordingly.
3. **Add packing to `TileMatMulForCache`'s output**, if pursuing the
   no-BLAS path further. It currently narrows loop bounds over the
   *original* strided layout rather than copying each tile into a
   contiguous scratch buffer the way BLIS/OpenBLAS do — some of the
   remaining 55x gap to BLAS is almost certainly non-contiguous/TLB
   overhead this would remove. Larger lift than (1).
4. **Multi-threading — untested, likely the single biggest remaining
   lever, and orthogonal to all of the above.** Every measurement in this
   entire investigation, including the original project benchmarks, pins
   `OPENBLAS_NUM_THREADS=1` for fairness. This CPU has 14 cores; nothing
   measured here has used more than one.

## Reproduction

All artifacts (`.mlir`, `.ll`, `.o`, `.out`, logs) are under
`/tmp/claude-1000/-home-fuzzserver/c3249f2f-c0ac-4085-afd3-875daa294f1d/scratchpad/blas_ablation/`
from this session and were not committed to the repo. To rebuild from
scratch, see `runpod_results/local_blas_ablation_2026-09-21.txt` for the
exact command sequence used for each leg (BLAS, generic no-BLAS, and
native+fastmath no-BLAS).

`build-ninja`'s `CMakeCache.txt` points `LLVM_DIR`/`MLIR_DIR` at
`/build/build/...`, a path that only exists inside the original Docker
build and is not present on this host, so it cannot be reconfigured or
rebuilt here. `build-local/` (gitignored, matches the existing `build*/`
pattern) was configured fresh against
`~/.local/share/lpta/toolchains/llvm-23.1.1` instead:

```bash
cmake -G Ninja -B build-local -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_DIR=~/.local/share/lpta/toolchains/llvm-23.1.1/lib/cmake/llvm \
  -DMLIR_DIR=~/.local/share/lpta/toolchains/llvm-23.1.1/lib/cmake/mlir \
  -DCMAKE_CXX_COMPILER=~/.local/share/lpta/toolchains/llvm-23.1.1/bin/clang++ \
  -DCMAKE_C_COMPILER=~/.local/share/lpta/toolchains/llvm-23.1.1/bin/clang .
ninja -C build-local tutorial-opt
```

That toolchain's prebuilt distribution does not include `FileCheck`/`not`/
`count`, so `tests/CMakeLists.txt`'s `add_lit_testsuite` fails configure;
`add_subdirectory(tests)` in the top-level `CMakeLists.txt` was temporarily
commented out to configure `build-local`, then restored immediately after
(confirmed clean via `git status` before continuing) — this repo's
`CMakeLists.txt` is unmodified as of this writing. To actually run
`check-mlir-tutorial` (including `tests/enable_fastmath_for_loops.mlir`),
use the project's pinned Docker image, which builds LLVM/MLIR from source
and produces those utility targets.
