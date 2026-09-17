#!/usr/bin/env python3
"""Runs `src/sample/model.py`'s `Sample` module four ways -- eager,
`torch.compile` with the default Inductor backend, `torch.compile` with this
project's own `mlir_tutorial_backend`, and a hand-written fused Triton kernel
(`src/triton_kernels/fused_linear_clamp.py`) -- and checks all four agree.
Then runs a second module with an op the MLIR backend doesn't recognize, to
show the eager-fallback path actually firing.

No single environment in this repo currently has both torch_mlir and
triton+CUDA together (the torch-mlir-dev container has the former, this
host's `dev/` venv has the latter, neither has a GPU), so the Triton path
degrades gracefully to a clearly-labeled skip rather than failing the whole
demo -- it will run for real once both are available together, e.g. on a
RunPod/Colab GPU session with torch_mlir built the way the rest of this
project already builds it there.

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

    return model, x, eager_out


def triton_path_demo(model, x, eager_out):
    print("\n=== Fourth comparison: hand-written fused Triton kernel ===")
    triton_dir = str(Path(__file__).resolve().parents[1] / "triton_kernels")
    if triton_dir not in sys.path:
        sys.path.insert(0, triton_dir)

    try:
        from fused_linear_clamp import fused_linear_clamp
    except SystemExit:
        print("[triton] triton/torch not importable here -> skipped "
              "(this container has torch_mlir but not triton; run this comparison "
              "from an environment with both, e.g. a RunPod/Colab GPU session)")
        return
    if not torch.cuda.is_available():
        print("[triton] no CUDA device in this environment -> skipped "
              "(kernel correctness is verified separately, without a GPU, via "
              "`TRITON_INTERPRET=1 python3 src/triton_kernels/fused_linear_clamp.py --interpret`)")
        return

    param = model.param.detach().cuda()
    weight = model.linear.weight.detach().cuda()
    bias = model.linear.bias.detach().cuda()

    def run(x_arg):
        return fused_linear_clamp(x_arg.cuda(), param, weight, bias).cpu()

    triton_out, triton_t = _time_call(run, x)
    torch.testing.assert_close(triton_out, eager_out, rtol=1e-3, atol=1e-4)
    print(f"triton   : {triton_t * 1e6:8.2f} us/call")
    print("output matches eager (hand-written fused Triton kernel correct)")


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
    model, x, eager_out = supported_path_demo()
    triton_path_demo(model, x, eager_out)
    fallback_path_demo()
