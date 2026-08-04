# L4 Evaluation Runbook

This runbook evaluates the GPU matmul, batch-matmul, attention, convolution,
and residual-convolution paths on one dedicated NVIDIA L4. Formal timings must
come from the same GPU and container. Do not run diagnostics, profilers, or
other GPU workloads concurrently with timing.

## Prepared Image

The evaluation image contains two isolated Python environments:

- `/opt/venv`: PyTorch 2.13 CPU and the pinned torch-mlir compiler.
- `/opt/pytorch-cuda`: PyTorch 2.8.0 with CUDA 12.8 for eager baselines.

The prepared public image is:

```text
priyatam19/mlir-matmul-blas-lowering:l4-eval-8da8a9b
priyatam19/mlir-matmul-blas-lowering@sha256:31b7d9addffac9e37a81c4dbb9a9e5efe533ee0e14d328400a1f5b02cd9ba1c5
```

It was built from commit `8da8a9bed553e336d5bb6500fe5b4d222abebc4c`.
Use the digest form as the RunPod container image. The pod should have at least
50 GB of container disk and a persistent `/workspace` volume of at least 20 GB.
Use one 24 GB L4 and expose a direct TCP SSH endpoint so the final archive can
be copied with SCP. The public image can be pulled without registry
credentials.

## Start and Resume

Clone the public repository, checkout the exact commit used to build the image,
and run the evaluation under `tmux`:

```bash
cd /workspace
git clone https://github.com/priyatam19/mlir-matmul-blas-lowering.git mlir_project
cd mlir_project
git checkout 8da8a9bed553e336d5bb6500fe5b4d222abebc4c

export RESULTS_DIR=/workspace/results/l4_eval_$(date -u +%Y%m%dT%H%M%SZ)
export EVALUATION_IMAGE_REF=priyatam19/mlir-matmul-blas-lowering@sha256:31b7d9addffac9e37a81c4dbb9a9e5efe533ee0e14d328400a1f5b02cd9ba1c5
tmux new -s l4-eval
EVAL_PHASE=all bash src/benchmarks/run_l4_evaluation.sh \
  2>&1 | tee "${RESULTS_DIR}/evaluation.log"
```

Each successful phase writes a marker under `${RESULTS_DIR}/.phases`. To
resume, reconnect to `tmux` or rerun with the same `RESULTS_DIR`; completed
phases are skipped. A single phase can be run explicitly:

```bash
RESULTS_DIR=/workspace/results/<existing-run> EVAL_PHASE=sanitizer \
  bash src/benchmarks/run_l4_evaluation.sh
```

The phases are `preflight`, `build`, `correctness`, `sanitizer`, `timing`,
`sweeps`, `profile`, and `archive`. The profiling phase uses Nsight Systems,
which does not require access to GPU hardware performance counters. Every
phase failure is blocking.

## Measurement Contract

- Preflight requires an idle L4 with at least 22,000 MiB visible memory and
  rejects a set `CUDA_LAUNCH_BLOCKING`.
- Every binary is built once before the GPU checks. The build runs the complete
  tutorial lit suite and generates `sm_89` code.
- Correctness compares every output element with cuBLAS, cuDNN, or a
  deterministic PyTorch reference and checks launch/API counts.
- Compute Sanitizer covers custom and vendor irregular BMM/convolution plus
  both mixed programs using correctness-only invocations.
- Formal timing uses 10 warmups, 50 samples, and three process-level trials.
  Mode order rotates between trials. No timed process runs concurrently.
- Tuning uses 5 warmups and 20 samples for four BMM tiles and four convolution
  thread counts. Nsight Systems runs only after formal measurements and traces
  one correctness invocation per representative mode.
- Profiling records untiled, custom, and vendor long BMM and ResNet convolution
  paths, plus custom and vendor attention and residual blocks. It exports
  timeline reports and CSV summaries for CUDA APIs, kernels, memory activity,
  launch latency, and OS runtime behavior.
- Canonical values are medians of the three trial p10, p50, and p90 values.

The summarizer enforces these performance gates:

- Long-attention custom BMM is at least 10x faster than untiled lowering.
- ResNet 3x3 custom convolution is at least 10x faster than untiled lowering.
- Every vendor BMM/convolution is within 5x of direct cuBLAS/cuDNN.
- PR2 `gemm_512` block-thread p50 remains between 0.1206 and 0.1474 ms.

## Failure and Recovery

If preflight reports the wrong GPU, insufficient memory, another compute
process, or unavailable CUDA, stop the pod immediately. If correctness or
sanitizer fails, do not time that mode. Keep the generated object, diagnostic,
reference, and sanitizer logs.

An incomplete run can still be packaged; missing gates make the archive phase
return nonzero after writing its files:

```bash
RESULTS_DIR=/workspace/results/<run> EVAL_PHASE=archive \
  bash src/benchmarks/run_l4_evaluation.sh || true
```

Copy the archive through the direct TCP endpoint before stopping the pod:

```bash
scp -P <port> -i ~/.ssh/id_ed25519 \
  root@<host>:/workspace/results/<run>.tar.gz .
sha256sum <run>.tar.gz
```

The archive contains the environment manifest, phase markers, build logs,
diagnostics, sanitizer output, raw CSV files, tuning results, profiler output,
`summary.csv`, `gates.json`, `evaluation_summary.md`, and per-file checksums.

The completed 2026-08-04 run is summarized in
[NVIDIA L4 Evaluation Results](l4_evaluation_results.md).
