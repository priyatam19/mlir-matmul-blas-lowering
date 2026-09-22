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

## Result 4: What's the ceiling? PyTorch eager vs. `torch.compile` (Inductor)

Every run of `src/benchmarks/run_real_benchmarks.sh` in this repo's history
has used `SKIP_PYTORCH=1` — the PyTorch reference point
(`src/benchmarks/pytorch_bench.py`, same four shapes as everywhere else in
this doc) has never actually been executed before. It was run for this
section, and a new `src/benchmarks/pytorch_compile_bench.py` adds
`torch.compile` (default Inductor backend) on top, both single-threaded
(matching every other number in this doc) and at PyTorch's own default
thread count (14 on this CPU).

p50, nanoseconds:

| Shape | Custom BLAS (notiled) | PyTorch eager, 1 thread | PyTorch eager, 14 threads | `torch.compile`, 1 thread |
|---|---:|---:|---:|---:|
| bert_attn | 47,879 | 22,375 | 8,875 | 35,253 |
| resnet_conv | 263,235 | 155,239 | 49,543 | 170,035 |
| gpt2_ffn | 424,161 | 303,057 | 93,012 | 322,228 |
| large_gemm | 1,641,752 | 1,212,554 | 299,044 | 1,229,799 |
| gemm2048 | 158,798,966 | 152,879,592 | 36,114,037 | 152,977,525 |

Three findings, in order of how surprising they are:

1. **`torch.compile`'s default backend does not help a bare matmul — it's
   1.00x-1.58x *slower* than eager**, worse the smaller the shape. This
   isn't a bug or a bad measurement: Inductor's value proposition is fusing
   *multiple* ops in a graph (eliminating intermediate materializations,
   fewer kernel launches). A lone `torch.matmul` has nothing to fuse —
   Inductor recognizes it as an "extern kernel" and calls out to the exact
   same ATen/BLAS implementation eager already calls, just with dispatch
   and guard overhead wrapped around it. So the "advanced backend" isn't
   doing anything smarter at the GEMM kernel level here; GEMM was never
   Inductor's kernel to begin with, in either mode. (Inductor's fusion
   advantage should show up on the *mixed* programs — attention block,
   residual conv — the same way it evidently doesn't on isolated GEMM;
   worth testing separately if useful.)
2. **PyTorch eager, at the same 1-thread constraint as everything else in
   this doc, is already 1.35x-2.14x faster than this project's custom
   BLAS-converted pipeline**, largest gap at the smallest shape. This is a
   *BLAS quality* gap, not a codegen gap: this PyTorch wheel links a
   better-tuned BLAS/oneDNN build than the generic `libopenblas-dev` apt
   package installed for this investigation (Results 1-3). Same lever
   category as "stop tiling before BLAS" from Result 3 — it's about which
   library gets called, not what MLIR generates.
3. **Multi-threading, confirmed as the single largest lever** (this doc's
   Next Steps previously listed it as untested/predicted-largest): going
   from 1 to 14 threads is worth 2.52x-4.23x on eager PyTorch alone, growing
   with problem size. Combined with (2), **PyTorch eager at its own default
   thread count is 4.4x-5.5x faster than this project's custom pipeline's
   best result (BLAS, notiled) at every shape tested** — using nothing more
   exotic than the ordinary multi-threaded BLAS call PyTorch already makes
   by default. That is the realistic ceiling on this hardware for a bare
   GEMM, and neither MLIR-level tiling nor `torch.compile` gets there;
   linking a better BLAS and actually using more than one thread does.

## Result 5: Does tiling really help at a scale multifold past every cache level, and does targeting L1 specifically matter?

Result 3's 2048^3 case (48 MB, ~2x the 24 MB L3) used a tile
(`128x128x256`, 320 KB of A+B+C) sized for the 1.25 MB **L2**, not L1. This
result goes further on both axes: `4096x4096x4096` (~64 MB per matrix,
~192 MB total working set — **8x** the L3, an unambiguous multifold-over-cache
case), and a second tile config explicitly sized for the 48 KB **L1**.

L1 sizing: for three square tiles of side `T` in fp32,
`3 * T^2 * 4 bytes <= 48 KB` gives `T <= 64` at the exact boundary (no
margin for anything else L1 holds — loop counters, spills). `tile-m=32
tile-n=32 tile-k=32` gives `3 * 32^2 * 4 = 12 KB`, a comfortable 25% of L1.
4096 divides evenly by both 32 and 128, so neither tiling needs tail
handling. 1 warmup + 2 timed calls per variant (each call is
30 s-12 min at this scale; variance between the 2 timed calls was under
0.3% for every variant, so 2 was enough to trust).

p50, and GFLOP/s (`2*4096^3 = 137.4 GFLOP`):

| Variant | BLAS | no-BLAS (native+fastmath) |
|---|---:|---:|
| untiled | 1.234 s (111.4 GFLOP/s) | 710.9 s (0.19 GFLOP/s) |
| L2-tile (320 KB) | 1.523 s (90.2 GFLOP/s), **+23.4%** | 70.1 s (1.96 GFLOP/s), **10.1x faster** |
| L1-tile (12 KB) | 2.628 s (52.3 GFLOP/s), **+112.9%** | 9.0 s (15.24 GFLOP/s), **78.9x faster** |

The two paths point in exactly opposite directions as the tile gets
smaller, and both are consistent with everything established so far:

- **With BLAS, smaller tiles are strictly worse, and get worse faster than
  at 2048^3.** L1-sized tiling means ~2.1 million separate `cblas_sgemm`
  calls (`(4096/32)^3`); at that call count, per-call overhead alone
  (parameter marshalling, BLAS-internal setup) dominates real compute.
  OpenBLAS already blocks internally — chopping its input into pieces this
  small only adds call-count tax with no locality benefit it didn't already
  have.
- **Without BLAS, smaller (properly cache-matched) tiles are dramatically
  better, and fitting the *fastest* cache level, not just *a* cache level,
  is most of the win.** Going from no tiling to an L2-fit tile is a real
  10.1x; going one step further, from L2-fit to L1-fit, is another 7.8x on
  top of that — nearly as large a jump as the first one. This is the clean,
  unambiguous "yes, tiling helps" result Result 3 pointed at but didn't
  fully demonstrate: it needed both a scale that actually exceeds every
  cache level (not just barely past L2, as the smaller real-world shapes
  and even 2048^3's own working set mostly were) and a tile matched to the
  cache level that actually matters most for reuse.

For scale: even the best custom result here (L1-tile, no-BLAS,
15.24 GFLOP/s) remains ~7.3x slower than plain untiled BLAS
(111.4 GFLOP/s) — this is evidence that *tiling itself* is a large,
real, correctly-targeted lever, not evidence that this custom pipeline
has caught up to a vendor BLAS. It hasn't; see Result 4 for how large
that remaining gap (and the threading/BLAS-choice gap on top of it) still
is.

## Result 6: Does packing close more of the gap?

Result 5 flagged that even the best custom result (L1-tile, no-BLAS,
15.24 GFLOP/s) trails untiled BLAS by ~7.3x, and attributed part of that to
`--tile-matmul-for-cache` narrowing loop bounds over the *original* strided
matrix rather than packing each tile into contiguous memory the way
BLIS/OpenBLAS do internally. This tests that directly: a new
`--pack-tiled-matmul-operands` pass, run after `--tile-matmul-for-cache` and
`--linalg-to-bufferization`, copies each tile's A/B operands into a fresh
contiguous buffer before the tile's `linalg.matmul` runs (C is left as-is;
it's written once per position, not the reuse-heavy operand).

**Two implementation attempts, both instructive.** The first tried to do
this at the *tensor* level, as a `pack=true` option directly inside
`--tile-matmul-for-cache`, using `bufferization.alloc_tensor` +
`tensor.insert_slice` right where each `tensor.extract_slice` tile was
produced. It compiled, ran, and produced correct output — but One-Shot
Bufferize recognized the copy as a same-size, zero-offset, unit-stride
tensor materialization (semantically a no-op) and **optimized it away
entirely**, silently leaving the original strided `memref.subview` in
place. The generated IR had zero extra buffers; the "packed" binary was
byte-for-byte the same computation as the unpacked one, and the ~3.5%
timing difference first measured was just noise, not a real effect. Lesson:
tensor-level "pack" attempts are subject to bufferization's own redundant-copy
elimination and can silently do nothing — verify by reading the bufferized
IR, not just by timing.

The second attempt moved packing to a **separate pass running after
bufferization** (`lib/PackTiledMatmulOperands.cpp`), operating directly on
`memref.subview` inputs to `linalg.matmul` — no tensor-level dataflow
analysis left at that point to see through it. This surfaced two more
issues before it worked: `memref.copy` lowers to a runtime call
(`memrefCopy`) this project's CPU path isn't linked against, and — more
importantly — a `memref.alloca` placed at the tile's own site, inside three
nested nested `scf.for` loops, is not hoisted by the backend across
iterations here; at cache-exceeding scale that's millions of stack frames
that are never popped, and it **segfaults from stack overflow** on the
first call. The fix for both: hand-roll the copy as an explicit
`scf.for`/`scf.for` load-store loop (avoids the runtime dependency, stays
inlinable and vectorizable), and explicitly hoist the `memref.alloca` to
just before the outermost enclosing `scf.for` so one buffer is allocated
once and reused every iteration — the same shape a real microkernel's
scratch space takes. Ragged-tail tiles (dynamic size, can't be computed
before the loop producing them exists) fall back to allocating at the
matmul's own site; every shape benchmarked in this document divides evenly
and never hits that path.

Correctness re-verified against the BLAS reference (checksum
`2671.086102` vs. `2671.086106`, FP32 rounding only) before trusting the
timing:

| Variant | p50 | GFLOP/s | vs. L1-tile w/o packing |
|---|---:|---:|---:|
| L1-tile, no packing (Result 5) | 9.018 s | 15.24 | — |
| L1-tile, **packed** | 5.452 s | 25.21 | **1.65x faster** |

Packing is a real, meaningful win on top of tiling — not a wash like the
first (silently-eliminated) attempt suggested. Stacked with tiling itself,
packed+tiled is now 130.4x faster than untiled (up from 78.9x), and closes
some more of the remaining gap to BLAS: 34.3x slower than untiled BLAS, down
from 55.4x. BLAS is still doing substantially more — multi-level (not just
single-level) blocking, SIMD-width-aware micro-kernels, and everything
Result 4 already covers — but packing alone recovered a genuine third of
the log-scale distance this specific lever had left on the table.

## Next Steps

Result 5's 78.9x is scale-dependent, not a general property of tiling: it
only shows up once the problem genuinely exceeds every cache level. None of
Result 1's real-world shapes (128 KB-2 MB working sets) are anywhere near
that regime, which is exactly why tiling looked irrelevant there and
decisive here — same pass, same mechanism, different scale.

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
2. **Stop tiling before BLAS. Done.** `docs/benchmark_results.md` no longer
   recommends chaining `--tile-matmul-for-cache` into
   `--convert-matmul-to-blas`.
3. **Add packing on top of tiling for the no-BLAS path. Done (Result 6)** —
   a real 1.65x on top of tiling's own 78.9x (130.4x total vs. untiled),
   via a new `--pack-tiled-matmul-operands` pass
   (`lib/PackTiledMatmulOperands.cpp`). Remaining gap to untiled BLAS is
   now 34.3x, down from 55.4x.
4. **Multi-threading — confirmed the single biggest lever (Result 4:
   2.52x-4.23x, growing with problem size), still untried in this custom
   pipeline.** Every MLIR-side measurement in this investigation pins
   `OPENBLAS_NUM_THREADS=1`/1 thread for fairness; nothing produced by this
   project's own passes has ever used more than one.
5. **Link a better BLAS.** Result 4 shows PyTorch's own linked
   BLAS/oneDNN beats the generic `libopenblas-dev` apt package by
   1.35x-2.14x single-threaded, on identical hardware and identical
   `cblas_sgemm`-shaped calls. This is a build/link-time choice, not an
   MLIR change, and it's free.
6. **`torch.compile`'s default backend is not a shortcut past any of
   this.** Result 4 shows it adds overhead over eager for an isolated
   matmul rather than helping — it has nothing to fuse. It may still be a
   useful ceiling reference for *mixed* programs (attention block,
   residual conv), where its fusion story is actually applicable; untested
   here.

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
