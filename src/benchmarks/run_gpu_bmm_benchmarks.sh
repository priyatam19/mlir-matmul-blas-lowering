#!/usr/bin/env bash
set -euo pipefail

PROJ="${PROJ:-/workspace/mlir_project}"
BUILD="${BUILD:-${PROJ}/build-ninja}"
MLIR_BUILD_DIR="${MLIR_BUILD_DIR:-/build/build}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CUDA_CHIP="${CUDA_CHIP:-sm_89}"
CUDA_PTX_FEATURE="${CUDA_PTX_FEATURE:-+ptx80}"
GPU_LOWERING="${GPU_LOWERING:-block-thread}"
BLOCK_M="${BLOCK_M:-8}"
BLOCK_N="${BLOCK_N:-32}"
SHAPES="${SHAPES:-bmm_bert bmm_long bmm_value bmm_irregular}"

if [[ "${GPU_LOWERING}" != "untiled" &&
      "${GPU_LOWERING}" != "block-thread" &&
      "${GPU_LOWERING}" != "vendor" ]]; then
  echo "Batch matmul benchmarks support GPU_LOWERING=untiled, block-thread, or vendor." >&2
  exit 2
fi

export PATH="${MLIR_BUILD_DIR}/bin:${BUILD}/tools:${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${MLIR_BUILD_DIR}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

declare -A models=(
  [bmm_bert]="gpu_bmm_bert.mlir"
  [bmm_long]="gpu_bmm_long.mlir"
  [bmm_value]="gpu_bmm_value.mlir"
  [bmm_irregular]="gpu_bmm_irregular.mlir"
)
declare -A harnesses=(
  [bmm_bert]="gpu_bmm_bert_bench.cpp"
  [bmm_long]="gpu_bmm_long_bench.cpp"
  [bmm_value]="gpu_bmm_value_bench.cpp"
  [bmm_irregular]="gpu_bmm_irregular_bench.cpp"
)

for shape in ${SHAPES}; do
  if [[ -z "${models[${shape}]:-}" ]]; then
    echo "Unknown batch matmul shape: ${shape}" >&2
    exit 2
  fi

  model="${PROJ}/src/benchmarks/${models[${shape}]}"
  harness="${PROJ}/src/benchmarks/${harnesses[${shape}]}"
  artifact_dir="${BUILD}/benchmark-artifacts/${shape}_${GPU_LOWERING}_${BLOCK_M}x${BLOCK_N}"
  mkdir -p "${artifact_dir}"
  binary="${artifact_dir}/${shape}.out"

  echo "Building ${shape}: lowering=${GPU_LOWERING} block=${BLOCK_M}x${BLOCK_N}" >&2
  (
    cd "${PROJ}/src/sample/gpu"
    MODEL_MLIR="${model}" GPU_LOWERING="${GPU_LOWERING}" \
      BLOCK_M="${BLOCK_M}" BLOCK_N="${BLOCK_N}" \
      CUDA_CHIP="${CUDA_CHIP}" CUDA_PTX_FEATURE="${CUDA_PTX_FEATURE}" \
      OUTPUT_DIR="${artifact_dir}" \
      bash run_mlir_pipeline.sh
    runtime=0
    if [[ "${GPU_LOWERING}" == "vendor" ]]; then
      runtime=1
    fi
    SAMPLE_CALL="${harness}" OUTPUT_BINARY="${binary}" LINK_CUBLAS=1 \
      LINK_TUTORIAL_GPU_RUNTIME="${runtime}" WRAP_MALLOC=0 \
      SAMPLE_OBJECT="${artifact_dir}/sample.o" \
      COMPILE_WORK_DIR="${artifact_dir}" bash compile.sh
  )

  if [[ "${BUILD_ONLY:-0}" == "1" ]]; then
    echo "Built ${binary}" >&2
    continue
  fi

  if [[ "${CHECK_LAUNCHES:-0}" == "1" ]]; then
    diagnostic_log="$(mktemp)"
    if [[ "${GPU_LOWERING}" == "vendor" ]]; then
      TUTORIAL_GPU_RUNTIME_DEBUG=1 LAUNCH_CHECK_ONLY=1 \
        GPU_LOWERING="${GPU_LOWERING}" "${binary}" 2>"${diagnostic_log}"
      call_count="$(grep -c 'op=cublas_batch_matmul' "${diagnostic_log}" || true)"
      echo "vendor_call_count=${call_count} expected=1 shape=${shape}" >&2
      [[ "${call_count}" == "1" ]]
    else
      MLIR_CUDA_DEBUG=1 LAUNCH_CHECK_ONLY=1 GPU_LOWERING="${GPU_LOWERING}" \
        "${binary}" 2>"${diagnostic_log}"
      launch_count="$(grep -c 'Launching kernel' "${diagnostic_log}" || true)"
      echo "launch_count=${launch_count} expected=1 shape=${shape}" >&2
      [[ "${launch_count}" == "1" ]]
    fi
    rm -f "${diagnostic_log}"
  fi

  result="$(GPU_LOWERING="${GPU_LOWERING}" BLOCK_M="${BLOCK_M}" \
    BLOCK_N="${BLOCK_N}" WARMUPS="${WARMUPS:-10}" RUNS="${RUNS:-50}" \
    "${binary}")"
  printf '%s\n' "${result}"
  if [[ -n "${RESULTS_FILE:-}" ]]; then
    printf '%s\n' "${result}" >>"${RESULTS_FILE}"
  fi
done
