#!/usr/bin/env bash
set -e
set -o pipefail

CONTAINER_NAME="${CONTAINER_NAME:-ml-compiler-exercise-cuda-dev}"
OUT_DIR="${OUT_DIR:-$PWD/colab_bundle}"

mkdir -p "${OUT_DIR}"

docker exec "${CONTAINER_NAME}" bash -lc \
  'tar czf /workspace/project/colab_compiler_stack.tar.gz /build/build /opt/venv'

tar \
  --exclude='./build-ninja' \
  --exclude='./colab_bundle' \
  --exclude='./colab_compiler_stack.tar.gz' \
  --exclude='./src/sample/*.o' \
  --exclude='./src/sample/*.out' \
  --exclude='./src/sample/*.ll' \
  --exclude='./src/sample/*_llvm*.mlir' \
  --exclude='./src/sample/gpu/*.o' \
  --exclude='./src/sample/gpu/*.ll' \
  --exclude='./src/sample/gpu/sample_gpu_dialect.mlir' \
  --exclude='./src/sample/gpu/sample_nvptx_isa.mlir' \
  -czf "${OUT_DIR}/ML-compiler-exercise-source.tar.gz" .

mv colab_compiler_stack.tar.gz "${OUT_DIR}/colab_compiler_stack.tar.gz"

cat <<EOF
Created:
  ${OUT_DIR}/colab_compiler_stack.tar.gz
  ${OUT_DIR}/ML-compiler-exercise-source.tar.gz

Upload both files to Google Drive before running the Colab trial notebook cells
from docs/ColabFreeGpuTrial.md.
EOF
