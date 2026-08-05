# L4 Evaluation Artifacts

This directory contains the compact outputs from the 2026-08-04 unified L4
evaluation:

- `environment.txt`: pinned software and GPU manifest.
- `summary.csv`: canonical timing and sweep aggregates.
- `gates.json`: machine-readable acceptance-gate results.
- `evaluation_summary.md`: generated timing and gate tables.
- `nsys_summary.md`: interpreted timeline, CUDA API, and CPU/GPU bottleneck
  findings from ten Nsight Systems traces.
- `nsys/`: machine-readable CUDA API, kernel, memory, launch-latency, and OS
  runtime CSV summaries. Interactive reports remain in the complete archive.

The complete raw archive was retrieved and verified locally as
`l4_eval_20260804T151913Z.tar.gz` with SHA-256
`56ac9fee61cc753b9087af2695e869e9d686585567411f835111f824d57e75b6`.
