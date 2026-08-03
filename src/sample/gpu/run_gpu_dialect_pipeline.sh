#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_DIALECT_ONLY=1 bash "${SCRIPT_DIR}/run_mlir_pipeline.sh"
