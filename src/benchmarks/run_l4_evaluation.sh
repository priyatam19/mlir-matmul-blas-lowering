#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="${PROJ:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
BUILD="${BUILD:-${PROJ}/build-ninja}"
MLIR_BUILD_DIR="${MLIR_BUILD_DIR:-/build/build}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
PYTORCH_CUDA_PYTHON="${PYTORCH_CUDA_PYTHON:-/opt/pytorch-cuda/bin/python}"
EVAL_PHASE="${EVAL_PHASE:-all}"
TRIALS="${TRIALS:-3}"
WARMUPS="${WARMUPS:-10}"
RUNS="${RUNS:-50}"
SWEEP_WARMUPS="${SWEEP_WARMUPS:-5}"
SWEEP_RUNS="${SWEEP_RUNS:-20}"
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"
REQUIRE_GPU_NAME="${REQUIRE_GPU_NAME:-NVIDIA L4}"
INTER_TRIAL_SLEEP="${INTER_TRIAL_SLEEP:-5}"
RESULTS_DIR="${RESULTS_DIR:-/workspace/results/l4_eval_$(date -u +%Y%m%dT%H%M%SZ)}"
MARKER_DIR="${RESULTS_DIR}/.phases"

export PATH="${MLIR_BUILD_DIR}/bin:${BUILD}/tools:${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${MLIR_BUILD_DIR}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_HOME
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

mkdir -p "${MARKER_DIR}" "${RESULTS_DIR}/build" "${RESULTS_DIR}/diagnostics" \
  "${RESULTS_DIR}/sanitizer" "${RESULTS_DIR}/raw" \
  "${RESULTS_DIR}/references" "${RESULTS_DIR}/sweeps" \
  "${RESULTS_DIR}/profiles"

log() {
  printf '[l4-eval] %s\n' "$*" >&2
}

run_logged() {
  local output=$1
  shift
  "$@" 2>&1 | tee "${output}"
}

require_marker() {
  local phase=$1
  if [[ ! -e "${MARKER_DIR}/${phase}.complete" ]]; then
    log "Phase '${phase}' must complete first."
    return 2
  fi
}

phase_preflight() {
  if [[ -n "${CUDA_LAUNCH_BLOCKING:-}" ]]; then
    log "CUDA_LAUNCH_BLOCKING must be unset for this evaluation."
    return 2
  fi
  command -v nvidia-smi >/dev/null
  command -v compute-sanitizer >/dev/null
  command -v nvcc >/dev/null
  test -x "${PYTORCH_CUDA_PYTHON}"
  test -x /opt/venv/bin/python

  local gpu_name memory_mib
  gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | xargs)"
  memory_mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | xargs)"
  if [[ "${gpu_name}" != *"${REQUIRE_GPU_NAME}"* ]]; then
    log "Expected ${REQUIRE_GPU_NAME}, found ${gpu_name}."
    return 2
  fi
  if (( memory_mib < 22000 )); then
    log "Expected an approximately 24 GiB L4, found ${memory_mib} MiB."
    return 2
  fi
  local active_processes
  active_processes="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader \
    | sed '/^[[:space:]]*$/d' | wc -l)"
  if (( active_processes != 0 )); then
    log "Expected an idle GPU, found ${active_processes} compute processes."
    return 2
  fi
  git -C "${PROJ}" diff --quiet
  git -C "${PROJ}" diff --cached --quiet

  {
    printf 'timestamp_utc=%s\n' "$(date -u +%FT%TZ)"
    printf 'git_sha=%s\n' "$(git -C "${PROJ}" rev-parse HEAD)"
    printf 'git_branch=%s\n' "$(git -C "${PROJ}" branch --show-current)"
    printf 'evaluation_image=%s\n' "${EVALUATION_IMAGE_REF:-unknown}"
    printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES}"
    printf 'gpu_name=%s\n' "${gpu_name}"
    printf 'gpu_memory_mib=%s\n' "${memory_mib}"
    printf 'cpu_count=%s\n' "$(nproc)"
    printf 'memory_kib=%s\n' "$(awk '/MemTotal/ {print $2}' /proc/meminfo)"
    printf 'compiler_python='; /opt/venv/bin/python -c 'import torch; print(torch.__version__)'
    printf 'cuda_python='; "${PYTORCH_CUDA_PYTHON}" -c 'import torch; print(torch.__version__)'
    printf 'cuda_python_cuda='; "${PYTORCH_CUDA_PYTHON}" -c 'import torch; print(torch.version.cuda)'
    printf 'cuda_available='; "${PYTORCH_CUDA_PYTHON}" -c 'import torch; print(torch.cuda.is_available())'
    printf 'cudnn='; "${PYTORCH_CUDA_PYTHON}" -c 'import torch; print(torch.backends.cudnn.version())'
    nvcc --version
    nvidia-smi --query-gpu=name,uuid,driver_version,pstate,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.total,memory.used --format=csv
    df -h "${RESULTS_DIR}"
  } >"${RESULTS_DIR}/environment.txt"
  "${PYTORCH_CUDA_PYTHON}" -c \
    'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
}

phase_build() {
  run_logged "${RESULTS_DIR}/build/configure.log" \
    cmake -S "${PROJ}" -B "${BUILD}" -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_DIR="${MLIR_BUILD_DIR}/lib/cmake/llvm" \
      -DMLIR_DIR="${MLIR_BUILD_DIR}/lib/cmake/mlir" \
      -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
      -DLLVM_USE_LINKER=lld
  run_logged "${RESULTS_DIR}/build/tutorial.log" \
    cmake --build "${BUILD}" --target tutorial-opt check-mlir-tutorial \
      -j"${BUILD_JOBS}"

  run_logged "${RESULTS_DIR}/build/gemm.log" env PROJ="${PROJ}" BUILD="${BUILD}" \
    BUILD_ONLY=1 GPU_LOWERING=block-thread SHAPES=gemm_512 BLOCK_M=8 BLOCK_N=32 \
    bash "${SCRIPT_DIR}/run_gpu_gemm_benchmarks.sh"

  local mode tile threads
  for mode in untiled block-thread vendor; do
    run_logged "${RESULTS_DIR}/build/bmm_${mode}.log" env PROJ="${PROJ}" BUILD="${BUILD}" \
      BUILD_ONLY=1 GPU_LOWERING="${mode}" BLOCK_M=8 BLOCK_N=32 \
      bash "${SCRIPT_DIR}/run_gpu_bmm_benchmarks.sh"
    run_logged "${RESULTS_DIR}/build/attention_${mode}.log" env PROJ="${PROJ}" BUILD="${BUILD}" \
      BUILD_ONLY=1 GPU_LOWERING="${mode}" \
      bash "${SCRIPT_DIR}/run_gpu_attention_benchmark.sh"
    run_logged "${RESULTS_DIR}/build/conv_${mode}.log" env PROJ="${PROJ}" BUILD="${BUILD}" \
      BUILD_ONLY=1 GPU_LOWERING="${mode}" CONV_THREADS=256 \
      bash "${SCRIPT_DIR}/run_gpu_conv_benchmarks.sh"
    run_logged "${RESULTS_DIR}/build/residual_${mode}.log" env PROJ="${PROJ}" BUILD="${BUILD}" \
      BUILD_ONLY=1 GPU_LOWERING="${mode}" \
      bash "${SCRIPT_DIR}/run_gpu_residual_conv_benchmark.sh"
  done

  for tile in 16x16 16x32 32x8; do
    run_logged "${RESULTS_DIR}/build/bmm_sweep_${tile}.log" env PROJ="${PROJ}" BUILD="${BUILD}" \
      BUILD_ONLY=1 GPU_LOWERING=block-thread SHAPES="bmm_bert bmm_long" \
      BLOCK_M="${tile%x*}" BLOCK_N="${tile#*x}" \
      bash "${SCRIPT_DIR}/run_gpu_bmm_benchmarks.sh"
  done
  for threads in 64 128 512; do
    run_logged "${RESULTS_DIR}/build/conv_sweep_t${threads}.log" env PROJ="${PROJ}" BUILD="${BUILD}" \
      BUILD_ONLY=1 GPU_LOWERING=block-thread \
      SHAPES="conv_resnet_block conv_pointwise" CONV_THREADS="${threads}" \
      bash "${SCRIPT_DIR}/run_gpu_conv_benchmarks.sh"
  done
}

generate_references() {
  local attention="${RESULTS_DIR}/references/attention.bin"
  local residual="${RESULTS_DIR}/references/residual.bin"
  ATTENTION_REFERENCE_OUT="${attention}" REFERENCE_ONLY=1 \
    /opt/venv/bin/python "${SCRIPT_DIR}/pytorch_attention_bench.py"
  RESIDUAL_CONV_REFERENCE_OUT="${residual}" REFERENCE_ONLY=1 \
    /opt/venv/bin/python "${SCRIPT_DIR}/pytorch_residual_conv_bench.py"
  [[ "$(stat -c %s "${attention}")" == "393216" ]]
  [[ "$(stat -c %s "${residual}")" == "65536" ]]
}

diagnose_binary() {
  local label=$1 mode=$2 binary=$3 expected_launches=$4 call_pattern=$5 expected_calls=$6
  shift 6
  local stdout="${RESULTS_DIR}/diagnostics/${label}.out"
  local stderr="${RESULTS_DIR}/diagnostics/${label}.log"
  env "$@" GPU_LOWERING="${mode}" MLIR_CUDA_DEBUG=1 \
    TUTORIAL_GPU_RUNTIME_DEBUG=1 LAUNCH_CHECK_ONLY=1 \
    "${binary}" >"${stdout}" 2>"${stderr}"
  grep -q '^launch_check,' "${stdout}"
  local launches calls
  launches="$(grep -c 'Launching kernel' "${stderr}" || true)"
  calls=0
  if [[ "${call_pattern}" != "-" ]]; then
    calls="$(grep -c "${call_pattern}" "${stderr}" || true)"
  fi
  printf 'launch_count=%s expected=%s vendor_calls=%s expected=%s\n' \
    "${launches}" "${expected_launches}" "${calls}" "${expected_calls}" \
    | tee -a "${stdout}"
  [[ "${launches}" == "${expected_launches}" && "${calls}" == "${expected_calls}" ]]
}

phase_correctness() {
  require_marker build
  generate_references
  local mode shape binary launches calls pattern

  binary="${PROJ}/src/sample/gpu/gemm_512_block-thread_8x32.out"
  diagnose_binary gemm_512_block-thread block-thread "${binary}" 1 - 0 BLOCK_M=8 BLOCK_N=32

  for mode in untiled block-thread vendor; do
    for shape in bmm_bert bmm_long bmm_value bmm_irregular; do
      binary="${BUILD}/benchmark-artifacts/${shape}_${mode}_8x32/${shape}.out"
      launches=1; calls=0; pattern=-
      if [[ "${mode}" == vendor ]]; then
        launches=0; calls=1; pattern=op=cublas_batch_matmul
      fi
      diagnose_binary "${shape}_${mode}" "${mode}" "${binary}" \
        "${launches}" "${pattern}" "${calls}" BLOCK_M=8 BLOCK_N=32
    done
    binary="${BUILD}/benchmark-artifacts/attention_${mode}/attention.out"
    launches=14; calls=0; pattern=-
    if [[ "${mode}" == vendor ]]; then
      launches=12; calls=2; pattern=op=cublas_batch_matmul
    fi
    diagnose_binary "attention_${mode}" "${mode}" "${binary}" \
      "${launches}" "${pattern}" "${calls}" \
      PYTORCH_REFERENCE="${RESULTS_DIR}/references/attention.bin"

    for shape in conv_small conv_resnet_stem conv_resnet_block conv_pointwise conv_irregular; do
      binary="${BUILD}/benchmark-artifacts/${shape}_${mode}_t256/${shape}.out"
      launches=1; calls=0; pattern=-
      if [[ "${mode}" == vendor ]]; then
        launches=0; calls=1; pattern=op=cudnn_conv2d
      fi
      diagnose_binary "${shape}_${mode}" "${mode}" "${binary}" \
        "${launches}" "${pattern}" "${calls}" CONV_THREADS=256
    done
    binary="${BUILD}/benchmark-artifacts/residual_conv_${mode}/residual_conv.out"
    launches=9; calls=0; pattern=-
    if [[ "${mode}" == vendor ]]; then
      launches=7; calls=2; pattern=op=cudnn_conv2d
    fi
    diagnose_binary "residual_${mode}" "${mode}" "${binary}" \
      "${launches}" "${pattern}" "${calls}" \
      PYTORCH_REFERENCE="${RESULTS_DIR}/references/residual.bin"
  done
}

sanitize_binary() {
  local label=$1 mode=$2 binary=$3
  shift 3
  env "$@" GPU_LOWERING="${mode}" LAUNCH_CHECK_ONLY=1 \
    compute-sanitizer --tool memcheck --target-processes all \
      --error-exitcode 99 "${binary}" \
      >"${RESULTS_DIR}/sanitizer/${label}.out" \
      2>"${RESULTS_DIR}/sanitizer/${label}.log"
  grep -q 'ERROR SUMMARY: 0 errors' \
    "${RESULTS_DIR}/sanitizer/${label}.out" \
    "${RESULTS_DIR}/sanitizer/${label}.log"
}

phase_sanitizer() {
  require_marker correctness
  local mode
  for mode in block-thread vendor; do
    sanitize_binary "bmm_irregular_${mode}" "${mode}" \
      "${BUILD}/benchmark-artifacts/bmm_irregular_${mode}_8x32/bmm_irregular.out" \
      BLOCK_M=8 BLOCK_N=32
    sanitize_binary "conv_irregular_${mode}" "${mode}" \
      "${BUILD}/benchmark-artifacts/conv_irregular_${mode}_t256/conv_irregular.out" \
      CONV_THREADS=256
    sanitize_binary "attention_${mode}" "${mode}" \
      "${BUILD}/benchmark-artifacts/attention_${mode}/attention.out" \
      PYTORCH_REFERENCE="${RESULTS_DIR}/references/attention.bin"
    sanitize_binary "residual_${mode}" "${mode}" \
      "${BUILD}/benchmark-artifacts/residual_conv_${mode}/residual_conv.out" \
      PYTORCH_REFERENCE="${RESULTS_DIR}/references/residual.bin"
  done
}

mode_order() {
  case $(( ($1 - 1) % 3 )) in
    0) echo "untiled block-thread vendor" ;;
    1) echo "vendor untiled block-thread" ;;
    2) echo "block-thread vendor untiled" ;;
  esac
}

capture_timing() {
  local family=$1 name=$2 mode=$3 trial=$4 binary=$5
  shift 5
  local output="${RESULTS_DIR}/raw/timing__${family}__${name}__${mode}__trial_${trial}.csv"
  local error="${output%.csv}.log"
  env "$@" GPU_LOWERING="${mode}" WARMUPS="${WARMUPS}" RUNS="${RUNS}" \
    "${binary}" >"${output}" 2>"${error}"
  grep -q '^result,' "${output}"
}

capture_pytorch() {
  local family=$1 name=$2 trial=$3 script=$4
  local output="${RESULTS_DIR}/raw/timing__${family}__${name}__pytorch__trial_${trial}.csv"
  env WARMUPS="${WARMUPS}" RUNS="${RUNS}" \
    "${PYTORCH_CUDA_PYTHON}" "${script}" >"${output}" 2>"${output%.csv}.log"
  grep -q '^result,' "${output}"
}

phase_timing() {
  require_marker sanitizer
  unset CUDA_LAUNCH_BLOCKING
  local trial mode shape binary
  for ((trial = 1; trial <= TRIALS; ++trial)); do
    log "Formal timing trial ${trial}/${TRIALS}"
    nvidia-smi --query-gpu=timestamp,name,pstate,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used \
      --format=csv >"${RESULTS_DIR}/raw/gpu_before_trial_${trial}.csv"

    capture_timing gemm gemm_512 block-thread "${trial}" \
      "${PROJ}/src/sample/gpu/gemm_512_block-thread_8x32.out" BLOCK_M=8 BLOCK_N=32

    for shape in bmm_bert bmm_long bmm_value bmm_irregular; do
      for mode in $(mode_order "${trial}"); do
        binary="${BUILD}/benchmark-artifacts/${shape}_${mode}_8x32/${shape}.out"
        capture_timing bmm "${shape}" "${mode}" "${trial}" "${binary}" \
          BLOCK_M=8 BLOCK_N=32
      done
    done
    for mode in $(mode_order "${trial}"); do
      capture_timing attention attention_block "${mode}" "${trial}" \
        "${BUILD}/benchmark-artifacts/attention_${mode}/attention.out" \
        PYTORCH_REFERENCE="${RESULTS_DIR}/references/attention.bin"
    done

    for shape in conv_small conv_resnet_stem conv_resnet_block conv_pointwise conv_irregular; do
      for mode in $(mode_order "${trial}"); do
        binary="${BUILD}/benchmark-artifacts/${shape}_${mode}_t256/${shape}.out"
        capture_timing conv "${shape}" "${mode}" "${trial}" "${binary}" \
          CONV_THREADS=256
      done
    done
    for mode in $(mode_order "${trial}"); do
      capture_timing residual residual_conv_block "${mode}" "${trial}" \
        "${BUILD}/benchmark-artifacts/residual_conv_${mode}/residual_conv.out" \
        PYTORCH_REFERENCE="${RESULTS_DIR}/references/residual.bin"
    done

    capture_pytorch attention attention_block "${trial}" \
      "${SCRIPT_DIR}/pytorch_attention_bench.py"
    capture_pytorch residual residual_conv_block "${trial}" \
      "${SCRIPT_DIR}/pytorch_residual_conv_bench.py"

    nvidia-smi --query-gpu=timestamp,name,pstate,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used \
      --format=csv >"${RESULTS_DIR}/raw/gpu_after_trial_${trial}.csv"
    sleep "${INTER_TRIAL_SLEEP}"
  done
}

capture_sweep() {
  local family=$1 name=$2 config=$3 binary=$4
  shift 4
  local output="${RESULTS_DIR}/sweeps/sweep__${family}__${name}__${config}__trial_1.csv"
  env "$@" GPU_LOWERING=block-thread WARMUPS="${SWEEP_WARMUPS}" \
    RUNS="${SWEEP_RUNS}" "${binary}" >"${output}" 2>"${output%.csv}.log"
  grep -q '^result,' "${output}"
}

phase_sweeps() {
  require_marker timing
  local tile shape threads
  for tile in 8x32 16x16 16x32 32x8; do
    for shape in bmm_bert bmm_long; do
      capture_sweep bmm "${shape}" "tile-${tile}" \
        "${BUILD}/benchmark-artifacts/${shape}_block-thread_${tile}/${shape}.out" \
        BLOCK_M="${tile%x*}" BLOCK_N="${tile#*x}"
    done
  done
  for threads in 64 128 256 512; do
    for shape in conv_resnet_block conv_pointwise; do
      capture_sweep conv "${shape}" "threads-${threads}" \
        "${BUILD}/benchmark-artifacts/${shape}_block-thread_t${threads}/${shape}.out" \
        CONV_THREADS="${threads}"
    done
  done
}

phase_profile() {
  require_marker sweeps
  if ! command -v ncu >/dev/null; then
    printf 'ncu unavailable; profiling skipped.\n' >"${RESULTS_DIR}/profiles/SKIPPED.txt"
    return 0
  fi
  local status=0
  env GPU_LOWERING=block-thread BLOCK_M=8 BLOCK_N=32 LAUNCH_CHECK_ONLY=1 \
    ncu --set basic --target-processes all --csv \
      --log-file "${RESULTS_DIR}/profiles/bmm_long.csv" \
      "${BUILD}/benchmark-artifacts/bmm_long_block-thread_8x32/bmm_long.out" \
      >"${RESULTS_DIR}/profiles/bmm_long.out" 2>"${RESULTS_DIR}/profiles/bmm_long.log" \
      || status=$?
  env GPU_LOWERING=block-thread CONV_THREADS=256 LAUNCH_CHECK_ONLY=1 \
    ncu --set basic --target-processes all --csv \
      --log-file "${RESULTS_DIR}/profiles/conv_resnet_block.csv" \
      "${BUILD}/benchmark-artifacts/conv_resnet_block_block-thread_t256/conv_resnet_block.out" \
      >"${RESULTS_DIR}/profiles/conv_resnet_block.out" \
      2>"${RESULTS_DIR}/profiles/conv_resnet_block.log" || status=$?
  if (( status != 0 )); then
    printf 'ncu returned %d; profiling is non-blocking.\n' "${status}" \
      >"${RESULTS_DIR}/profiles/SKIPPED.txt"
  fi
  return 0
}

phase_archive() {
  local summary_status=0
  /opt/venv/bin/python "${SCRIPT_DIR}/summarize_l4_evaluation.py" \
    --results-dir "${RESULTS_DIR}" --require-passing || summary_status=$?
  (
    cd "${RESULTS_DIR}"
    find . -type f ! -name SHA256SUMS -print0 | sort -z \
      | xargs -0 sha256sum >SHA256SUMS
  )
  local archive="${RESULTS_DIR}.tar.gz"
  tar -czf "${archive}" -C "$(dirname "${RESULTS_DIR}")" \
    "$(basename "${RESULTS_DIR}")"
  log "Results archive: ${archive}"
  return "${summary_status}"
}

run_phase() {
  local phase=$1
  local marker="${MARKER_DIR}/${phase}.complete"
  if [[ -e "${marker}" ]]; then
    log "Skipping completed phase: ${phase}"
    return 0
  fi
  log "Starting phase: ${phase}"
  "phase_${phase}"
  date -u +%FT%TZ >"${marker}"
  log "Completed phase: ${phase}"
}

case "${EVAL_PHASE}" in
  all)
    for phase in preflight build correctness sanitizer timing sweeps profile archive; do
      run_phase "${phase}"
    done
    ;;
  preflight|build|correctness|sanitizer|timing|sweeps|profile|archive)
    run_phase "${EVAL_PHASE}"
    ;;
  *)
    log "Unknown EVAL_PHASE=${EVAL_PHASE}."
    exit 2
    ;;
esac
