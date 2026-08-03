#!/usr/bin/env python3
"""Export and benchmark a mixed-operation attention block on CUDA."""

import os
import statistics
import time

import torch
from torch_mlir import fx
from torch_mlir.compiler_utils import OutputType


class AttentionBlock(torch.nn.Module):
    def forward(self, query, key, value):
        scores = torch.bmm(query, key.transpose(1, 2)) * 0.125
        probabilities = torch.softmax(scores, dim=-1)
        return torch.bmm(probabilities, value)


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[int((len(ordered) - 1) * fraction)]


def main():
    torch.manual_seed(41)
    export_module = AttentionBlock().eval()
    cpu_inputs = (
        torch.randn(12, 128, 64),
        torch.randn(12, 128, 64),
        torch.randn(12, 128, 64),
    )
    output_path = os.getenv("ATTENTION_MLIR_OUT")
    if output_path:
        mlir_module = fx.export_and_import(
            export_module,
            *cpu_inputs,
            output_type=OutputType.LINALG_ON_TENSORS,
            func_name="attention_block",
        )
        with open(output_path, "w", encoding="utf-8") as output:
            output.write(str(mlir_module))
        if os.getenv("EXPORT_ONLY") == "1":
            return

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    warmups = int(os.getenv("WARMUPS", "10"))
    runs = int(os.getenv("RUNS", "50"))
    batch, sequence, head_dim = 12, 128, 64
    module = AttentionBlock().eval().cuda()
    query = torch.randn(batch, sequence, head_dim, device="cuda")
    key = torch.randn(batch, sequence, head_dim, device="cuda")
    value = torch.randn(batch, sequence, head_dim, device="cuda")

    for _ in range(warmups):
        module(query, key, value)
    torch.cuda.synchronize()

    device_times = []
    wall_times = []
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    for _ in range(runs):
        torch.cuda.synchronize()
        start.record()
        wall_start = time.perf_counter()
        module(query, key, value)
        stop.record()
        stop.synchronize()
        wall_times.append((time.perf_counter() - wall_start) * 1000.0)
        device_times.append(start.elapsed_time(stop))

    print("kind,name,backend,runs,p10_ms,p50_ms,p90_ms,wall_p50_ms")
    print(
        "result,attention_block,pytorch-eager-cuda,"
        f"{runs},{percentile(device_times, 0.10):.9f},"
        f"{percentile(device_times, 0.50):.9f},"
        f"{percentile(device_times, 0.90):.9f},"
        f"{statistics.median(wall_times):.9f}"
    )

if __name__ == "__main__":
    main()
