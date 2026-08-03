#!/usr/bin/env python3
"""Export and benchmark a mixed-operation residual convolution block."""

import os
import statistics
import time

import torch
import torch.nn.functional as functional
from torch_mlir import fx
from torch_mlir.compiler_utils import OutputType


class ResidualConvBlock(torch.nn.Module):
    def forward(self, value, weight1, bias1, weight2, bias2):
        hidden = functional.pad(value, (1, 1, 1, 1))
        hidden = functional.conv2d(hidden, weight1, bias1)
        hidden = torch.relu(hidden)
        hidden = functional.pad(hidden, (1, 1, 1, 1))
        hidden = functional.conv2d(hidden, weight2, bias2)
        return torch.relu(hidden + value)


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[int((len(ordered) - 1) * fraction)]


def main():
    module = ResidualConvBlock().eval()
    value_elements = 16 * 32 * 32
    weight_elements = 16 * 16 * 3 * 3
    cpu_inputs = (
        ((torch.arange(value_elements) % 101 - 50).float() * 0.002).reshape(
            1, 16, 32, 32
        ),
        ((torch.arange(weight_elements) + 11) % 97 - 48)
        .float()
        .mul(0.001)
        .reshape(16, 16, 3, 3),
        (torch.arange(16).float() - 8).mul(0.001),
        ((torch.arange(weight_elements) + 29) % 89 - 44)
        .float()
        .mul(0.001)
        .reshape(16, 16, 3, 3),
        (7 - torch.arange(16).float()).mul(0.001),
    )
    output_path = os.getenv("RESIDUAL_CONV_MLIR_OUT")
    if output_path:
        mlir_module = fx.export_and_import(
            module,
            *cpu_inputs,
            output_type=OutputType.LINALG_ON_TENSORS,
            func_name="residual_conv_block",
        )
        with open(output_path, "w", encoding="utf-8") as output:
            output.write(str(mlir_module))
        if os.getenv("EXPORT_ONLY") == "1":
            return

    reference_path = os.getenv("RESIDUAL_CONV_REFERENCE_OUT")
    if reference_path:
        with torch.no_grad():
            module(*cpu_inputs).contiguous().numpy().tofile(reference_path)
        if os.getenv("REFERENCE_ONLY") == "1":
            return

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    warmups = int(os.getenv("WARMUPS", "10"))
    runs = int(os.getenv("RUNS", "50"))
    cuda_inputs = tuple(value.cuda() for value in cpu_inputs)
    module = module.cuda()

    for _ in range(warmups):
        module(*cuda_inputs)
    torch.cuda.synchronize()

    device_times = []
    wall_times = []
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    for _ in range(runs):
        torch.cuda.synchronize()
        start.record()
        wall_start = time.perf_counter()
        module(*cuda_inputs)
        stop.record()
        stop.synchronize()
        wall_times.append((time.perf_counter() - wall_start) * 1000.0)
        device_times.append(start.elapsed_time(stop))

    print("kind,name,backend,runs,p10_ms,p50_ms,p90_ms,wall_p50_ms")
    print(
        "result,residual_conv_block,pytorch-eager-cuda,"
        f"{runs},{percentile(device_times, 0.10):.9f},"
        f"{percentile(device_times, 0.50):.9f},"
        f"{percentile(device_times, 0.90):.9f},"
        f"{statistics.median(wall_times):.9f}"
    )


if __name__ == "__main__":
    main()
