# GPU Batch-Matmul and Attention Benchmarks

## Lowering Modes

PR 3 extends the post-bufferization GPU pipeline to rank-3 f32
`linalg.batch_matmul` operations with standard indexing maps:

| Mode | Batch matmul behavior |
|---|---|
| `untiled` | Generic linalg parallel-loop lowering |
| `block-thread` | Batch to grid Z, M/N tiles to grid Y/X, N-contiguous threads |
| `vendor` | One cached `cublasSgemmStridedBatched` runtime call |

The custom pass supports static, dynamic, identity, and strided memrefs. It
preserves `C += A*B`, carries K in an SSA accumulator, guards partial M/N
tiles, and emits one launch for the complete batch. Broadcasted or transposed
indexing maps remain on generic lowering.

The vendor pass casts supported operands to one canonical dynamic-strided ABI.
The runtime honors memref offsets, row strides, and batch strides, validates
unit innermost strides, and caches a pedantic-FP32 cuBLAS handle.

## Benchmark Programs

| Name | Batch | M | K | N | Workload |
|---|---:|---:|---:|---:|---|
| `bmm_bert` | 12 | 128 | 64 | 128 | BERT attention scores |
| `bmm_long` | 12 | 512 | 64 | 512 | Long-sequence attention |
| `bmm_value` | 32 | 128 | 64 | 64 | Attention value projection |
| `bmm_irregular` | 7 | 129 | 65 | 127 | Boundary handling |

Each operator benchmark uses managed preallocated rank-3 memrefs, prefetches
them to the GPU, resets C outside the timed interval, and compares every output
element against pedantic-FP32 cuBLAS. It reports CUDA-event and wall-clock
p10/p50/p90, GFLOP/s, and maximum absolute/relative error.

The PyTorch-derived `AttentionBlock` adds a transpose, scaling, stable softmax,
and a second batch matmul. Torch-MLIR produces two `linalg.batch_matmul`
operations and six generic elementwise/reduction operations. The complete
program compiles in all three modes, proving unmatched operations retain the
generic path.

## L4 Procedure

Build and launch the pinned CUDA image on an NVIDIA L4, then run:

```bash
cd /workspace/mlir_project

for mode in untiled block-thread vendor; do
  GPU_LOWERING="${mode}" CHECK_LAUNCHES=1 \
    RESULTS_FILE="bmm_${mode}.csv" \
    bash src/benchmarks/run_gpu_bmm_benchmarks.sh
done

for mode in untiled block-thread vendor; do
  GPU_LOWERING="${mode}" \
    bash src/benchmarks/run_gpu_attention_benchmark.sh
done

ATTENTION_MLIR_OUT=src/benchmarks/gpu_attention_block.mlir \
  python3 src/benchmarks/pytorch_attention_bench.py
```

`CHECK_LAUNCHES=1` expects one MLIR CUDA launch for `untiled` and
`block-thread`, or one runtime cuBLAS call for `vendor`. Diagnostics run outside
the timed samples. Do not use `CUDA_LAUNCH_BLOCKING=1` while timing.

## Offline Verification

- All four shapes compile and link in all three modes.
- The mixed attention program compiles through GPU dialect, `sm_89` NVPTX,
  LLVM IR, and final linking in all three modes.
- Structural lowering produces one custom launch with N mapped to thread X.
- Vendor lowering replaces the two attention batch matmuls while preserving 12
  generic attention kernels.
- `check-mlir-tutorial` passes 9/9 tests.

Runtime measurements remain pending until a new L4 pod is available. The
custom long-attention kernel must beat generic lowering by at least 10x. Every
vendor-lowered operator must remain within 5x of its direct cuBLAS p50, and all
operator and attention outputs must pass full-output validation.
