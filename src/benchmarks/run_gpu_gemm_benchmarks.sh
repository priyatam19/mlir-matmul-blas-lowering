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
SHAPES="${SHAPES:-gemm_512 gemm_1024 gemm_2048 gemm_mobilenet gemm_irregular gemm_4096 gemm_tall gemm_wide gemm_ktail}"

export PATH="${MLIR_BUILD_DIR}/bin:${BUILD}/tools:${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${MLIR_BUILD_DIR}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

declare -A models=(
  [gemm_512]="gpu_gemm_512.mlir"
  [gemm_1024]="gpu_gemm_1024.mlir"
  [gemm_2048]="gpu_gemm_2048.mlir"
  [gemm_mobilenet]="gpu_gemm_mobilenet.mlir"
  [gemm_irregular]="gpu_gemm_irregular.mlir"
  [gemm_4096]="gpu_gemm_4096.mlir"
  [gemm_tall]="gpu_gemm_tall.mlir"
  [gemm_wide]="gpu_gemm_wide.mlir"
  [gemm_ktail]="gpu_gemm_ktail.mlir"
)
declare -A harnesses=(
  [gemm_512]="large_gemm_gpu_bench.cpp"
  [gemm_1024]="gpu_gemm_1024_bench.cpp"
  [gemm_2048]="gpu_gemm_2048_bench.cpp"
  [gemm_mobilenet]="gpu_gemm_mobilenet_bench.cpp"
  [gemm_irregular]="gpu_gemm_irregular_bench.cpp"
  [gemm_4096]="gpu_gemm_4096_bench.cpp"
  [gemm_tall]="gpu_gemm_tall_bench.cpp"
  [gemm_wide]="gpu_gemm_wide_bench.cpp"
  [gemm_ktail]="gpu_gemm_ktail_bench.cpp"
)

for shape in ${SHAPES}; do
  if [[ -z "${models[${shape}]:-}" ]]; then
    echo "Unknown benchmark shape: ${shape}" >&2
    exit 2
  fi
  if [[ "${GPU_LOWERING}" == "legacy" && "${shape}" != "gemm_512" ]]; then
    echo "Skipping ${shape}: the expensive pre-tiled legacy fixture is only defined for gemm_512." >&2
    continue
  fi

  model="${PROJ}/src/benchmarks/${models[${shape}]}"
  if [[ "${GPU_LOWERING}" == "legacy" ]]; then
    model="${PROJ}/src/benchmarks/gpu_gemm_512_legacy.mlir"
  fi
  harness="${PROJ}/src/benchmarks/${harnesses[${shape}]}"
  artifact_dir="${BUILD}/benchmark-artifacts/${shape}_${GPU_LOWERING}_${BLOCK_M}x${BLOCK_N}"
  mkdir -p "${artifact_dir}"
  binary="${PROJ}/src/sample/gpu/${shape}_${GPU_LOWERING}_${BLOCK_M}x${BLOCK_N}.out"

  echo "Building ${shape}: lowering=${GPU_LOWERING} block=${BLOCK_M}x${BLOCK_N}" >&2
  (
    cd "${PROJ}/src/sample/gpu"
    MODEL_MLIR="${model}" GPU_LOWERING="${GPU_LOWERING}" \
      BLOCK_M="${BLOCK_M}" BLOCK_N="${BLOCK_N}" \
      CUDA_CHIP="${CUDA_CHIP}" CUDA_PTX_FEATURE="${CUDA_PTX_FEATURE}" \
      OUTPUT_DIR="${artifact_dir}" \
      bash run_mlir_pipeline.sh
    autotune_runtime=0
    if [[ "${GPU_LOWERING}" == "autotuned" ]]; then
      autotune_runtime=1
    fi
    SAMPLE_CALL="${harness}" OUTPUT_BINARY="${binary}" LINK_CUBLAS=1 WRAP_MALLOC=0 \
      LINK_TUTORIAL_AUTOTUNE_RUNTIME="${autotune_runtime}" \
      SAMPLE_OBJECT="${artifact_dir}/sample.o" \
      COMPILE_WORK_DIR="${artifact_dir}" bash compile.sh
  )

  if [[ "${BUILD_ONLY:-0}" == "1" ]]; then
    echo "Built ${binary}" >&2
    continue
  fi

  if [[ "${CHECK_LAUNCHES:-0}" == "1" ]]; then
    launch_log="$(mktemp)"
    MLIR_CUDA_DEBUG=1 LAUNCH_CHECK_ONLY=1 GPU_LOWERING="${GPU_LOWERING}" \
      BLOCK_M="${BLOCK_M}" BLOCK_N="${BLOCK_N}" \
      "${binary}" 2>"${launch_log}"
    launch_count="$(grep -c 'Launching kernel' "${launch_log}" || true)"
    rm -f "${launch_log}"
    expected=1
    if [[ "${GPU_LOWERING}" == "legacy" ]]; then
      expected=1024
    fi
    echo "launch_count=${launch_count} expected=${expected} shape=${shape}" >&2
    [[ "${launch_count}" == "${expected}" ]]
  fi

  result="$(GPU_LOWERING="${GPU_LOWERING}" BLOCK_M="${BLOCK_M}" \
    BLOCK_N="${BLOCK_N}" WARMUPS="${WARMUPS:-10}" RUNS="${RUNS:-50}" \
    "${binary}")"
  printf '%s\n' "${result}"
  if [[ -n "${RESULTS_FILE:-}" ]]; then
    printf '%s\n' "${result}" >>"${RESULTS_FILE}"
  fi
done
