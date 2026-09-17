#!/usr/bin/env python3
"""Runs `src/sample/model.py`'s `Sample` module three ways -- eager,
`torch.compile` with the default Inductor backend, and `torch.compile` with
this project's own `mlir_tutorial_backend` -- and checks all three agree.
Then runs a second module with an op the MLIR backend doesn't recognize, to
show the eager-fallback path actually firing.

Run inside the torch-mlir-dev container (needs torch_mlir + tutorial-opt):
    docker exec mlir-backend-dev python3 /workspace/project/src/torch_compile/demo.py
"""

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sample"))
from mlir_backend import mlir_tutorial_backend  # noqa: E402
from model import Sample  # noqa: E402


def _time_call(fn, x, runs=20):
    for _ in range(3):
        fn(x)
    start = time.perf_counter()
    for _ in range(runs):
        out = fn(x)
    elapsed = (time.perf_counter() - start) / runs
    return out, elapsed


def supported_path_demo():
    print("=== Supported subgraph: Sample (add -> linear -> clamp) ===")
    torch.manual_seed(41)
    model = Sample()
    x = torch.randn(3, 4)

    eager_out, eager_t = _time_call(model, x)

    inductor_model = torch.compile(model)
    inductor_out, inductor_t = _time_call(inductor_model, x)

    mlir_model = torch.compile(model, backend=mlir_tutorial_backend)
    mlir_out, mlir_t = _time_call(mlir_model, x)

    print(f"eager    : {eager_t * 1e6:8.2f} us/call")
    print(f"inductor : {inductor_t * 1e6:8.2f} us/call")
    print(f"mlir     : {mlir_t * 1e6:8.2f} us/call")

    torch.testing.assert_close(inductor_out, eager_out, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(mlir_out, eager_out, rtol=1e-4, atol=1e-5)
    print("outputs match eager (inductor and mlir_tutorial both correct)")


class Unsupported(nn.Module):
    def forward(self, x):
        return torch.sin(x)  # not in mlir_backend's allow-list


def fallback_path_demo():
    print("\n=== Unsupported subgraph: torch.sin -> expect eager fallback ===")
    model = Unsupported()
    x = torch.randn(3, 4)
    compiled = torch.compile(model, backend=mlir_tutorial_backend)
    out = compiled(x)
    torch.testing.assert_close(out, torch.sin(x))
    print("fell back to eager and produced the correct result")


if __name__ == "__main__":
    supported_path_demo()
    fallback_path_demo()
