# Tensor-Parallel Placement Analysis

This project has no distributed-memory or tensor-placement work anywhere in
it, and no multi-GPU hardware to build any. This doc closes part of that gap
the way the rest of the project already closes similar gaps before running
anything on real hardware: with a closed-form cost model, the same kind of
estimate `lib/LowerContractionToGpu.cpp`'s `estimateRegistersPerThread()`
uses to prune kernel candidates analytically before the empirical runtime
autotuner (`src/runtime/TutorialAutotuneRuntime.cpp`) ever measures anything.
It also extends this project's own roofline methodology
(`docs/deep_dive_part3_qa6to10.md`, Q76/Q81) one level up the memory
hierarchy: instead of asking whether a kernel is compute- or
DRAM-bandwidth-bound on one chip, this asks whether a tensor-parallel split
is compute- or interconnect-bound across multiple chips.

Script: `src/benchmarks/tensor_parallel_placement_analysis.py`. Everything
below is its actual printed output, not hand-computed or fabricated numbers.

## Setup

**Workload:** a transformer FFN/MLP block -- up-projection
(`d_model -> d_ffn`), GELU, down-projection (`d_ffn -> d_model`) -- split
using the standard Megatron-LM tensor-parallel pattern:

- **Up-projection is column-parallel**: shard `W1` by output column
  (`d_ffn` dimension) across GPUs. Each GPU computes a disjoint slice of the
  intermediate activation and applies GELU to it locally. No communication
  needed here -- GELU is elementwise, so slicing before it and applying it
  after doesn't change the result.
- **Down-projection is row-parallel**: shard `W2` by input row (also the
  `d_ffn` dimension), so each GPU's local slice of the intermediate feeds
  directly into its local slice of `W2` with no reshuffling. Each GPU now
  holds a partial sum of the full `(batch, d_model)` output.
- **One all-reduce** sums those partial `(batch, d_model)` outputs across
  GPUs to produce the real result. This is the entire communication cost of
  a Megatron MLP block -- everything else is embarrassingly parallel.

**Shapes:** this repo's own benchmarked GPT-2 FFN shape
(`src/benchmarks/gpt2_ffn.mlir`: `batch=128, d_model=256, d_ffn=512`), plus
real GPT-2-small dimensions (`d_model=768, d_ffn=3072`) for comparison, since
the repo's benchmark shape is intentionally scaled down and that turns out to
matter a great deal (see below).

**Hardware:** this project's actual target GPU, the NVIDIA L4
(`docs/l4_evaluation_results.md`: `sm_89`, driver 570.195.03, CUDA 12.8.1).
Compute peaks (30.3 TFLOPS FP32, 120 TFLOPS TF32) are the vendor datasheet
figures for that card, used as idealized roofs exactly the way the existing
roofline doc uses idealized CPU peaks -- not measured by this repo. The
interconnect assumption is **PCIe Gen4 x16 (64 GB/s)**, the same bandwidth
this project's own roofline doc already cites for host-device transfer. This
isn't a simplifying assumption picked for convenience: the L4 has no NVLink,
so for a mesh built from this exact hardware, PCIe is the only inter-GPU link
that exists. Communication cost uses a standard ring all-reduce model
(`2*(N-1)/N * payload / bandwidth`), which is itself an idealized lower bound
(no per-message latency floor, no link contention).

## Results

```
=== gpt2_ffn (this repo's benchmark): batch=128 d_model=256 d_ffn=512 ===
total FLOPs (both matmuls, forward only): 67.11 MFLOPs
all-reduce payload per block: 131072 bytes (128.0 KiB)
  N   math   compute/shard (us)    all-reduce (us)   compute/comm ratio         regime
  1   fp32                2.215              0.000                  n/a  compute-bound
  1   tf32                0.559              0.000                  n/a  compute-bound
  2   fp32                1.107              2.048                0.541     comm-bound
  2   tf32                0.280              2.048                0.137     comm-bound
  4   fp32                0.554              3.072                0.180     comm-bound
  4   tf32                0.140              3.072                0.046     comm-bound
  8   fp32                0.277              3.584                0.077     comm-bound
  8   tf32                0.070              3.584                0.020     comm-bound

=== gpt2-small (real dimensions): batch=128 d_model=768 d_ffn=3072 ===
total FLOPs (both matmuls, forward only): 1207.96 MFLOPs
all-reduce payload per block: 393216 bytes (384.0 KiB)
  N   math   compute/shard (us)    all-reduce (us)   compute/comm ratio         regime
  1   fp32               39.867              0.000                  n/a  compute-bound
  1   tf32               10.066              0.000                  n/a  compute-bound
  2   fp32               19.933              6.144                3.244  compute-bound
  2   tf32                5.033              6.144                0.819     comm-bound
  4   fp32                9.967              9.216                1.081  compute-bound
  4   tf32                2.517              9.216                0.273     comm-bound
  8   fp32                4.983             10.752                0.463     comm-bound
  8   tf32                1.258             10.752                0.117     comm-bound
```

## What this actually shows

1. **This repo's own benchmark shape is comm-bound at every tensor-parallel
   degree tested**, including `N=2`. The all-reduce payload (128 KiB) is
   small in absolute terms, but the matmuls are smaller still relative to
   it -- there just isn't enough compute per shard to hide a PCIe transfer
   behind.

2. **Real GPT-2-small dimensions flip this at low `N`**: at `N=2`/FP32 the
   block is compute-bound by more than 3x, and even at `N=4` it's roughly
   balanced (ratio 1.08). The only thing that changed between the two tables
   is `d_ffn` (512 -> 3072) and `d_model` (256 -> 768) -- the workload got
   wider. That's the general, not-shape-specific reason tensor parallelism
   works better on larger models: the all-reduce payload depends only on
   `(batch, d_model)`, but the compute depends on `(batch, d_model, d_ffn)`
   divided across shards -- a wider FFN buys more compute per byte moved
   without changing the communication cost at all.

3. **Switching to faster math makes communication relatively worse, not
   better.** TF32 cuts compute time by ~4x at every `N` in both tables, but
   the all-reduce cost doesn't move -- so the same GPT-2-small shape that was
   comfortably compute-bound in FP32 at `N=4` (ratio 1.08) becomes
   comm-bound in TF32 at the same `N` (ratio 0.27). This is the direct
   multi-GPU analogue of a point this project's own `LowerContractionToGpu`
   pass already had to reason about on a single chip: a faster arithmetic
   mode only helps if something else isn't now the bottleneck.

4. **More GPUs make the communication cost worse in absolute terms, not just
   relatively.** The all-reduce time strictly increases with `N`
   (`2*(N-1)/N` grows toward 2 as `N` grows) at the same time each shard's
   compute shrinks toward zero -- so past some `N`, adding GPUs to this
   split is a pure loss. This is the quantitative version of "tensor
   parallelism doesn't scale indefinitely," and it's why real systems bound
   the tensor-parallel degree to what fits in one NVLink-connected node and
   reach for pipeline or expert parallelism (different communication
   pattern, different bottleneck) to scale further.

## Honest limitations

- This models one isolated MLP block. A real transformer layer also has an
  attention block with its own tensor-parallel split and all-reduce, and a
  full model has many layers -- some of that communication can overlap with
  independent compute elsewhere (a different micro-batch, a different
  pipeline stage); none of that overlap is modeled here.
- The ring all-reduce model is an idealized lower bound: no fixed per-message
  latency, no contention from other traffic sharing the PCIe root complex,
  no accounting for the actual NCCL algorithm selection (small messages
  often use a different algorithm than large ones).
- FP32/TF32 compute figures are vendor peak, not this project's own measured
  achieved FLOPs (which the existing L4 evaluation shows land well below
  peak) -- so the real compute/comm ratios are somewhat worse than shown
  here, on both sides of every table.
