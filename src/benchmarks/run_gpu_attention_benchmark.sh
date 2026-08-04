#!/usr/bin/env bash
set -euo pipefail

PROJ="${PROJ:-/workspace/mlir_project}"
BUILD="${BUILD:-${PROJ}/build-ninja}"
MLIR_BUILD_DIR="${MLIR_BUILD_DIR:-/build/build}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CUDA_CHIP="${CUDA_CHIP:-sm_89}"
CUDA_PTX_FEATURE="${CUDA_PTX_FEATURE:-+ptx80}"
GPU_LOWERING="${GPU_LOWERING:-block-thread}"
MODEL="${PROJ}/src/benchmarks/gpu_attention_block.mlir"
HARNESS="${PROJ}/src/benchmarks/gpu_attention_block_bench.cpp"
ARTIFACT_DIR="${BUILD}/benchmark-artifacts/attention_${GPU_LOWERING}"
BINARY="${ARTIFACT_DIR}/attention.out"

mkdir -p "${ARTIFACT_DIR}"
export PATH="${MLIR_BUILD_DIR}/bin:${BUILD}/tools:${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${MLIR_BUILD_DIR}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

(
  cd "${PROJ}/src/sample/gpu"
  MODEL_MLIR="${MODEL}" GPU_LOWERING="${GPU_LOWERING}" \
    CUDA_CHIP="${CUDA_CHIP}" CUDA_PTX_FEATURE="${CUDA_PTX_FEATURE}" \
    OUTPUT_DIR="${ARTIFACT_DIR}" bash run_mlir_pipeline.sh
  runtime=0
  if [[ "${GPU_LOWERING}" == "vendor" ]]; then
    runtime=1
  fi
  SAMPLE_CALL="${HARNESS}" SAMPLE_OBJECT="${ARTIFACT_DIR}/sample.o" \
    COMPILE_WORK_DIR="${ARTIFACT_DIR}" OUTPUT_BINARY="${BINARY}" \
    LINK_TUTORIAL_GPU_RUNTIME="${runtime}" LINK_CUBLAS=1 WRAP_MALLOC=1 \
    bash compile.sh
)

if [[ "${BUILD_ONLY:-0}" == "1" ]]; then
  echo "Built ${BINARY}" >&2
  exit 0
fi

PYTORCH_REFERENCE="${ARTIFACT_DIR}/pytorch_reference.bin"
ATTENTION_REFERENCE_OUT="${PYTORCH_REFERENCE}" REFERENCE_ONLY=1 \
  python3 "${PROJ}/src/benchmarks/pytorch_attention_bench.py"
export PYTORCH_REFERENCE

if [[ "${CHECK_LAUNCHES:-0}" == "1" ]]; then
  diagnostic_log="$(mktemp)"
  if [[ "${GPU_LOWERING}" == "vendor" ]]; then
    MLIR_CUDA_DEBUG=1 TUTORIAL_GPU_RUNTIME_DEBUG=1 LAUNCH_CHECK_ONLY=1 \
      GPU_LOWERING="${GPU_LOWERING}" "${BINARY}" 2>"${diagnostic_log}"
    launch_count="$(grep -c 'Launching kernel' "${diagnostic_log}" || true)"
    call_count="$(grep -c 'op=cublas_batch_matmul' "${diagnostic_log}" || true)"
    echo "launch_count=${launch_count} expected=12 attention_block" >&2
    echo "vendor_call_count=${call_count} expected=2 attention_block" >&2
    [[ "${launch_count}" == "12" && "${call_count}" == "2" ]]
  else
    MLIR_CUDA_DEBUG=1 LAUNCH_CHECK_ONLY=1 GPU_LOWERING="${GPU_LOWERING}" \
      "${BINARY}" 2>"${diagnostic_log}"
    launch_count="$(grep -c 'Launching kernel' "${diagnostic_log}" || true)"
    echo "launch_count=${launch_count} expected=14 attention_block" >&2
    [[ "${launch_count}" == "14" ]]
  fi
  rm -f "${diagnostic_log}"
fi

GPU_LOWERING="${GPU_LOWERING}" WARMUPS="${WARMUPS:-5}" RUNS="${RUNS:-20}" \
  "${BINARY}"
