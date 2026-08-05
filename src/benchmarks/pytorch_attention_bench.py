#!/usr/bin/env python3
"""Export and benchmark a mixed-operation attention block on CUDA."""

import os
import statistics
import time

import torch


class AttentionBlock(torch.nn.Module):
    def forward(self, query, key, value):
        scores = torch.bmm(query, key.transpose(1, 2)) * 0.125
        probabilities = torch.softmax(scores, dim=-1)
        return torch.bmm(probabilities, value)


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[int((len(ordered) - 1) * fraction)]


def main():
    export_module = AttentionBlock().eval()
    elements = 12 * 128 * 64
    indices = torch.arange(elements)
    cpu_inputs = (
        ((indices % 101) + 1).float().mul(0.001).reshape(12, 128, 64),
        (((indices + 11) % 103) + 1).float().mul(0.001).reshape(12, 128, 64),
        (((indices + 17) % 107) + 1).float().mul(0.001).reshape(12, 128, 64),
    )
    output_path = os.getenv("ATTENTION_MLIR_OUT")
    if output_path:
        from torch_mlir import fx
        from torch_mlir.compiler_utils import OutputType

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

    reference_path = os.getenv("ATTENTION_REFERENCE_OUT")
    if reference_path:
        with torch.no_grad():
            export_module(*cpu_inputs).contiguous().numpy().tofile(reference_path)
        if os.getenv("REFERENCE_ONLY") == "1":
            return

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    warmups = int(os.getenv("WARMUPS", "10"))
    runs = int(os.getenv("RUNS", "50"))
    module = AttentionBlock().eval().cuda()
    query, key, value = (tensor.cuda() for tensor in cpu_inputs)

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
