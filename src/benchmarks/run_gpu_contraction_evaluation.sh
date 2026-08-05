#!/usr/bin/env bash
set -euo pipefail

PROJ="${PROJ:-/workspace/mlir_project}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJ}/runpod_results/shared_memory_gpu}"
TRIALS="${TRIALS:-3}"
WARMUPS="${WARMUPS:-10}"
RUNS="${RUNS:-50}"
GEMM_SHAPES="${GEMM_SHAPES:-gemm_512 gemm_1024 gemm_2048 gemm_mobilenet gemm_irregular gemm_4096 gemm_tall gemm_wide gemm_ktail}"
BMM_SHAPES="${BMM_SHAPES:-bmm_bert bmm_long bmm_value bmm_irregular}"
CONV_SHAPES="${CONV_SHAPES:-conv_small conv_resnet_stem conv_resnet_block conv_pointwise conv_irregular}"
mkdir -p "${OUTPUT_DIR}"

export TUTORIAL_AUTOTUNE_CACHE="${OUTPUT_DIR}/gpu-autotune-v1.csv"
{
  date --iso-8601=seconds
  nvidia-smi
  nvcc --version
  git -C "${PROJ}" rev-parse HEAD
} >"${OUTPUT_DIR}/environment.txt"

run_family() {
  local family="$1"
  local mode="$2"
  local shapes="$3"
  local output="$4"
  : >"${output}"
  case "${family}" in
    gemm)
      PROJ="${PROJ}" GPU_LOWERING="${mode}" SHAPES="${shapes}" \
        AUTOTUNE_MATH_MODE="${AUTOTUNE_MATH_MODE:-shared-fp32}" \
        GPU_MATH_MODE="${GPU_MATH_MODE:-fp32}" \
        WARMUPS="${WARMUPS}" RUNS="${RUNS}" RESULTS_FILE="${output}" \
        bash "${PROJ}/src/benchmarks/run_gpu_gemm_benchmarks.sh"
      ;;
    bmm)
      PROJ="${PROJ}" GPU_LOWERING="${mode}" SHAPES="${shapes}" \
        AUTOTUNE_MATH_MODE="${AUTOTUNE_MATH_MODE:-shared-fp32}" \
        GPU_MATH_MODE="${GPU_MATH_MODE:-fp32}" \
        WARMUPS="${WARMUPS}" RUNS="${RUNS}" RESULTS_FILE="${output}" \
        bash "${PROJ}/src/benchmarks/run_gpu_bmm_benchmarks.sh"
      ;;
    conv)
      PROJ="${PROJ}" GPU_LOWERING="${mode}" SHAPES="${shapes}" \
        AUTOTUNE_MATH_MODE="${AUTOTUNE_MATH_MODE:-shared-fp32}" \
        GPU_MATH_MODE="${GPU_MATH_MODE:-fp32}" \
        WARMUPS="${WARMUPS}" RUNS="${RUNS}" RESULTS_FILE="${output}" \
        bash "${PROJ}/src/benchmarks/run_gpu_conv_benchmarks.sh"
      ;;
  esac
}

if [[ "${SKIP_SWEEP:-0}" != "1" ]]; then
  PROJ="${PROJ}" RESULTS_FILE="${OUTPUT_DIR}/profile_sweep.csv" \
    WARMUPS="${WARMUPS}" RUNS="${RUNS}" \
    bash "${PROJ}/src/benchmarks/sweep_gpu_contraction_profiles.sh"
fi

for family in gemm bmm conv; do
  case "${family}" in
    gemm) shapes="${GEMM_SHAPES}" ;;
    bmm) shapes="${BMM_SHAPES}" ;;
    conv) shapes="${CONV_SHAPES}" ;;
  esac
  for mode in block-thread shared-fp32 tensorcore-tf32 vendor; do
    for trial in $(seq 1 "${TRIALS}"); do
      run_family "${family}" "${mode}" "${shapes}" \
        "${OUTPUT_DIR}/${family}_${mode}_trial${trial}.csv"
    done
  done
done

rm -f "${TUTORIAL_AUTOTUNE_CACHE}"
for family in gemm bmm conv; do
  case "${family}" in
    gemm) shapes="${GEMM_SHAPES}" ;;
    bmm) shapes="${BMM_SHAPES}" ;;
    conv) shapes="${CONV_SHAPES}" ;;
  esac
  WARMUPS=56 RUNS=1 run_family "${family}" autotuned "${shapes}" \
    "${OUTPUT_DIR}/${family}_autotune_cold.csv"
  for trial in $(seq 1 "${TRIALS}"); do
    run_family "${family}" autotuned "${shapes}" \
      "${OUTPUT_DIR}/${family}_autotuned_trial${trial}.csv"
  done
done

for family in gemm bmm conv; do
  case "${family}" in
    gemm) shapes="${GEMM_SHAPES}" ;;
    bmm) shapes="${BMM_SHAPES}" ;;
    conv) shapes="${CONV_SHAPES}" ;;
  esac
  AUTOTUNE_MATH_MODE=tensorcore-tf32 GPU_MATH_MODE=tf32 \
    WARMUPS=56 RUNS=1 run_family "${family}" autotuned "${shapes}" \
      "${OUTPUT_DIR}/${family}_autotune_tf32_cold.csv"
  for trial in $(seq 1 "${TRIALS}"); do
    AUTOTUNE_MATH_MODE=tensorcore-tf32 GPU_MATH_MODE=tf32 \
      run_family "${family}" autotuned "${shapes}" \
        "${OUTPUT_DIR}/${family}_autotuned_tf32_trial${trial}.csv"
  done
done

for mode in shared-fp32 tensorcore-tf32 autotuned; do
  PROJ="${PROJ}" GPU_LOWERING="${mode}" WARMUPS="${WARMUPS}" RUNS="${RUNS}" \
    bash "${PROJ}/src/benchmarks/run_gpu_attention_benchmark.sh" \
    >"${OUTPUT_DIR}/attention_${mode}.csv"
  PROJ="${PROJ}" GPU_LOWERING="${mode}" WARMUPS="${WARMUPS}" RUNS="${RUNS}" \
    bash "${PROJ}/src/benchmarks/run_gpu_residual_conv_benchmark.sh" \
    >"${OUTPUT_DIR}/residual_${mode}.csv"
done

AUTOTUNE_MATH_MODE=tensorcore-tf32 GPU_MATH_MODE=tf32 \
  PROJ="${PROJ}" GPU_LOWERING=autotuned WARMUPS="${WARMUPS}" RUNS="${RUNS}" \
  bash "${PROJ}/src/benchmarks/run_gpu_attention_benchmark.sh" \
  >"${OUTPUT_DIR}/attention_autotuned_tf32.csv"
AUTOTUNE_MATH_MODE=tensorcore-tf32 GPU_MATH_MODE=tf32 \
  PROJ="${PROJ}" GPU_LOWERING=autotuned WARMUPS="${WARMUPS}" RUNS="${RUNS}" \
  bash "${PROJ}/src/benchmarks/run_gpu_residual_conv_benchmark.sh" \
  >"${OUTPUT_DIR}/residual_autotuned_tf32.csv"

nvptx_mlir="${PROJ}/src/sample/gpu/sample_nvptx_isa.mlir"
if [[ -f "${nvptx_mlir}" ]]; then
  python3 "${PROJ}/src/benchmarks/extract_ptx.py" "${nvptx_mlir}" \
    "${OUTPUT_DIR}/ptx"
  : >"${OUTPUT_DIR}/ptxas_resources.txt"
  for ptx in "${OUTPUT_DIR}"/ptx/*.ptx; do
    cubin="${ptx%.ptx}.cubin"
    {
      echo "== ${ptx} =="
      ptxas --verbose --gpu-name sm_89 "${ptx}" --output-file "${cubin}"
    } 2>>"${OUTPUT_DIR}/ptxas_resources.txt"
  done
fi

if command -v nsys >/dev/null 2>&1; then
  binary="${PROJ}/src/sample/gpu/gemm_1024_tensorcore-tf32_8x32.out"
  if [[ -x "${binary}" ]]; then
    nsys profile --force-overwrite=true --trace=cuda,nvtx,cublas,osrt \
      --sample=none --cpuctxsw=none -o "${OUTPUT_DIR}/tf32_gemm_1024" \
      "${binary}"
  fi
fi

BUILD="${BUILD:-${PROJ}/build-ninja}"
declare -a sanitizer_binaries=(
  "${PROJ}/src/sample/gpu/gemm_irregular_autotuned_8x32.out"
  "${BUILD}/benchmark-artifacts/bmm_irregular_autotuned_8x32/bmm_irregular.out"
  "${BUILD}/benchmark-artifacts/conv_irregular_autotuned_t256/conv_irregular.out"
)
if command -v compute-sanitizer >/dev/null 2>&1; then
  : >"${OUTPUT_DIR}/compute_sanitizer.txt"
  for binary in "${sanitizer_binaries[@]}"; do
    if [[ -x "${binary}" ]]; then
      TUTORIAL_AUTOTUNE_READ_ONLY=1 LAUNCH_CHECK_ONLY=1 \
        compute-sanitizer --tool memcheck --error-exitcode=99 "${binary}" \
        >>"${OUTPUT_DIR}/compute_sanitizer.txt" 2>&1
    fi
  done
fi

python3 "${PROJ}/src/benchmarks/analyze_contraction_results.py" "${OUTPUT_DIR}"

echo "Evaluation artifacts: ${OUTPUT_DIR}"
