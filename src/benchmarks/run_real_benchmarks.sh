#!/usr/bin/env bash
set -euo pipefail

PROJ="${PROJ:-/workspace/project}"
BUILD=$PROJ/build-ninja
BENCH=$PROJ/src/benchmarks
TUTOPT=$BUILD/tools/tutorial-opt
OPENBLAS=/usr/lib/x86_64-linux-gnu/openblas-pthread

export LD_LIBRARY_PATH=$OPENBLAS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "================================================================"
echo "  Real-World GEMM Benchmark Suite"
echo "  Tiled MLIR+BLAS  vs  Non-tiled MLIR+BLAS  vs  PyTorch"
echo "================================================================"
echo ""
echo "Cache: L1d=48KB  L2=1.25MB  L3=24MB  |  OMP_NUM_THREADS=1"
echo ""

cd $BENCH

echo "[1/3] Building all binaries..."

# bert_attn: full=128KB, tile=(32x32)=20KB, FLOPs=2.1M
$TUTOPT --linalg-to-bufferization bert_attn.mlir > bert_attn_notiled_buf.mlir
$TUTOPT --llvm-request-c-wrappers --bufferization-to-llvm bert_attn_notiled_buf.mlir > bert_attn_notiled_llvm.mlir
mlir-translate -mlir-to-llvmir bert_attn_notiled_llvm.mlir > bert_attn_notiled.ll
llc --filetype=obj -O3 bert_attn_notiled.ll -o bert_attn_notiled.o
g++ -O2 -c bert_attn_bench.cpp -o bert_attn_bench.o
g++ -no-pie bert_attn_bench.o bert_attn_notiled.o -o bert_attn_notiled.out -L$OPENBLAS -lopenblas -lm
echo "  Built bert_attn non-tiled"
$TUTOPT --tile-matmul-for-cache='tile-m=32 tile-n=32 tile-k=64' --linalg-to-bufferization bert_attn.mlir > bert_attn_tiled_buf.mlir
$TUTOPT --llvm-request-c-wrappers --bufferization-to-llvm bert_attn_tiled_buf.mlir > bert_attn_tiled_llvm.mlir
mlir-translate -mlir-to-llvmir bert_attn_tiled_llvm.mlir > bert_attn_tiled.ll
llc --filetype=obj -O3 bert_attn_tiled.ll -o bert_attn_tiled.o
g++ -O2 -c bert_attn_bench.cpp -o bert_attn_bench_t.o
g++ -no-pie bert_attn_bench_t.o bert_attn_tiled.o -o bert_attn_tiled.out -L$OPENBLAS -lopenblas -lm
echo "  Built bert_attn tiled (tile-m=32 tile-n=32 tile-k=64)"

# resnet_conv: full=512KB, tile=(64x64)=80KB, FLOPs=16.8M
$TUTOPT --linalg-to-bufferization resnet_conv.mlir > resnet_conv_notiled_buf.mlir
$TUTOPT --llvm-request-c-wrappers --bufferization-to-llvm resnet_conv_notiled_buf.mlir > resnet_conv_notiled_llvm.mlir
mlir-translate -mlir-to-llvmir resnet_conv_notiled_llvm.mlir > resnet_conv_notiled.ll
llc --filetype=obj -O3 resnet_conv_notiled.ll -o resnet_conv_notiled.o
g++ -O2 -c resnet_conv_bench.cpp -o resnet_conv_bench.o
g++ -no-pie resnet_conv_bench.o resnet_conv_notiled.o -o resnet_conv_notiled.out -L$OPENBLAS -lopenblas -lm
echo "  Built resnet_conv non-tiled"
$TUTOPT --tile-matmul-for-cache='tile-m=64 tile-n=64 tile-k=128' --linalg-to-bufferization resnet_conv.mlir > resnet_conv_tiled_buf.mlir
$TUTOPT --llvm-request-c-wrappers --bufferization-to-llvm resnet_conv_tiled_buf.mlir > resnet_conv_tiled_llvm.mlir
mlir-translate -mlir-to-llvmir resnet_conv_tiled_llvm.mlir > resnet_conv_tiled.ll
llc --filetype=obj -O3 resnet_conv_tiled.ll -o resnet_conv_tiled.o
g++ -O2 -c resnet_conv_bench.cpp -o resnet_conv_bench_t.o
g++ -no-pie resnet_conv_bench_t.o resnet_conv_tiled.o -o resnet_conv_tiled.out -L$OPENBLAS -lopenblas -lm
echo "  Built resnet_conv tiled (tile-m=64 tile-n=64 tile-k=128)"

# gpt2_ffn: full=896KB, tile=(64x64)=144KB, FLOPs=33.6M
$TUTOPT --linalg-to-bufferization gpt2_ffn.mlir > gpt2_ffn_notiled_buf.mlir
$TUTOPT --llvm-request-c-wrappers --bufferization-to-llvm gpt2_ffn_notiled_buf.mlir > gpt2_ffn_notiled_llvm.mlir
mlir-translate -mlir-to-llvmir gpt2_ffn_notiled_llvm.mlir > gpt2_ffn_notiled.ll
llc --filetype=obj -O3 gpt2_ffn_notiled.ll -o gpt2_ffn_notiled.o
g++ -O2 -c gpt2_ffn_bench.cpp -o gpt2_ffn_bench.o
g++ -no-pie gpt2_ffn_bench.o gpt2_ffn_notiled.o -o gpt2_ffn_notiled.out -L$OPENBLAS -lopenblas -lm
echo "  Built gpt2_ffn non-tiled"
$TUTOPT --tile-matmul-for-cache='tile-m=64 tile-n=64 tile-k=256' --linalg-to-bufferization gpt2_ffn.mlir > gpt2_ffn_tiled_buf.mlir
$TUTOPT --llvm-request-c-wrappers --bufferization-to-llvm gpt2_ffn_tiled_buf.mlir > gpt2_ffn_tiled_llvm.mlir
mlir-translate -mlir-to-llvmir gpt2_ffn_tiled_llvm.mlir > gpt2_ffn_tiled.ll
llc --filetype=obj -O3 gpt2_ffn_tiled.ll -o gpt2_ffn_tiled.o
g++ -O2 -c gpt2_ffn_bench.cpp -o gpt2_ffn_bench_t.o
g++ -no-pie gpt2_ffn_bench_t.o gpt2_ffn_tiled.o -o gpt2_ffn_tiled.out -L$OPENBLAS -lopenblas -lm
echo "  Built gpt2_ffn tiled (tile-m=64 tile-n=64 tile-k=256)"

# large_gemm: full=2048KB, tile=(128x128)=320KB, FLOPs=134.2M
$TUTOPT --linalg-to-bufferization large_gemm.mlir > large_gemm_notiled_buf.mlir
$TUTOPT --llvm-request-c-wrappers --bufferization-to-llvm large_gemm_notiled_buf.mlir > large_gemm_notiled_llvm.mlir
mlir-translate -mlir-to-llvmir large_gemm_notiled_llvm.mlir > large_gemm_notiled.ll
llc --filetype=obj -O3 large_gemm_notiled.ll -o large_gemm_notiled.o
g++ -O2 -c large_gemm_bench.cpp -o large_gemm_bench.o
g++ -no-pie large_gemm_bench.o large_gemm_notiled.o -o large_gemm_notiled.out -L$OPENBLAS -lopenblas -lm
echo "  Built large_gemm non-tiled"
$TUTOPT --tile-matmul-for-cache='tile-m=128 tile-n=128 tile-k=256' --linalg-to-bufferization large_gemm.mlir > large_gemm_tiled_buf.mlir
$TUTOPT --llvm-request-c-wrappers --bufferization-to-llvm large_gemm_tiled_buf.mlir > large_gemm_tiled_llvm.mlir
mlir-translate -mlir-to-llvmir large_gemm_tiled_llvm.mlir > large_gemm_tiled.ll
llc --filetype=obj -O3 large_gemm_tiled.ll -o large_gemm_tiled.o
g++ -O2 -c large_gemm_bench.cpp -o large_gemm_bench_t.o
g++ -no-pie large_gemm_bench_t.o large_gemm_tiled.o -o large_gemm_tiled.out -L$OPENBLAS -lopenblas -lm
echo "  Built large_gemm tiled (tile-m=128 tile-n=128 tile-k=256)"

if [[ "${SKIP_PYTORCH:-0}" == "1" ]]; then
  echo "[2/3] PyTorch baseline skipped (SKIP_PYTORCH=1)."
  echo ""
else
  echo "[2/3] PyTorch baseline (single-threaded)..."
  echo ""
  python3 $BENCH/pytorch_bench.py
  echo ""
fi

echo "[3/3] MLIR benchmarks..."
echo ""
echo "── BERT attn head     (M=128 K=64  N=128) ──────────────────────────"
echo "   FLOPs=2.1M | full=128KB | tile(32x32)=20KB"
printf "   non-tiled (1 sgemm):          "; ./bert_attn_notiled.out
printf "   tiled     (tile-m=32 tile-n=32): "; ./bert_attn_tiled.out
echo ""
echo "── ResNet-50 conv3a   (M=256 K=128 N=256) ──────────────────────────"
echo "   FLOPs=16.8M | full=512KB | tile(64x64)=80KB"
printf "   non-tiled (1 sgemm):          "; ./resnet_conv_notiled.out
printf "   tiled     (tile-m=64 tile-n=64): "; ./resnet_conv_tiled.out
echo ""
echo "── GPT-2 FFN expand   (M=128 K=256 N=512) ──────────────────────────"
echo "   FLOPs=33.6M | full=896KB | tile(64x64)=144KB"
printf "   non-tiled (1 sgemm):          "; ./gpt2_ffn_notiled.out
printf "   tiled     (tile-m=64 tile-n=64): "; ./gpt2_ffn_tiled.out
echo ""
echo "── Large GEMM QKV     (M=512 K=256 N=512) ──────────────────────────"
echo "   FLOPs=134.2M | full=2048KB | tile(128x128)=320KB"
printf "   non-tiled (1 sgemm):          "; ./large_gemm_notiled.out
printf "   tiled     (tile-m=128 tile-n=128): "; ./large_gemm_tiled.out
echo ""
echo "================================================================"
echo ""
echo "Key: tiling forces sub-tile data to stay in L1/L2 between"
echo "the outer scf.for loop body iterations. For large_gemm, full"
echo "matrices (2MB) spill L2, while tiles (320KB) stay in L2."
echo "================================================================"
