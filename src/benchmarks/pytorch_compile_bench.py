#!/usr/bin/env python3
"""torch.compile (default Inductor backend) vs. eager, for the same shapes
used throughout docs/cpu_matmul_blas_ablation.md and pytorch_bench.py --
the "what's the achievable potential on this hardware" reference point for
that ablation, using PyTorch's own advanced compiler stack instead of this
project's custom MLIR pipeline or its custom TorchDynamo backend
(src/torch_compile/mlir_backend.py).

Runs each shape single-threaded (apples-to-apples with every other number
in the ablation, which pins OMP_NUM_THREADS=1/OPENBLAS_NUM_THREADS=1) and
at PyTorch's own default thread count (this machine's physical core count),
since multi-threading is an explicitly untested lever in that ablation.
"""
import statistics
import time

import torch

configs = [
    ("bert_attn",   128,  64, 128, 500),
    ("resnet_conv", 256, 128, 256, 200),
    ("gpt2_ffn",    128, 256, 512, 300),
    ("large_gemm",  512, 256, 512, 100),
    ("gemm2048",   2048, 2048, 2048, 15),
]


def matmul(a, b):
    return torch.matmul(a, b)


def time_call(fn, *args, warmup=20, runs=100):
    for _ in range(warmup):
        fn(*args)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter_ns()
        fn(*args)
        t1 = time.perf_counter_ns()
        times.append(t1 - t0)
    times.sort()
    avg = int(statistics.mean(times))
    p50 = times[len(times) // 2]
    p10 = times[len(times) // 10]
    p90 = times[len(times) * 9 // 10]
    return avg, p50, p10, p90


def fmt(label, M, K, N, avg, p50, p10, p90):
    flop = 2 * M * K * N
    gflops = flop / (p50 * 1e-9) / 1e9
    print(f"  {label:<28} avg={avg:>12} ns  p50={p50:>12}  p10={p10:>12}  "
          f"p90={p90:>12}  ({gflops:8.1f} GFLOP/s)")


def run(num_threads_label, num_threads=None):
    if num_threads is not None:
        torch.set_num_threads(num_threads)
    print(f"\n=== {num_threads_label} (torch.get_num_threads()="
          f"{torch.get_num_threads()}) ===")

    for name, M, K, N, runs in configs:
        A = torch.randn(M, K)
        B = torch.randn(K, N)

        eager_stats = time_call(matmul, A, B, runs=runs)
        fmt(f"{name} eager", M, K, N, *eager_stats)

        # torch.compile's default backend is Inductor. dynamic=False lets it
        # specialize the generated kernel to this exact static shape, same
        # as every MLIR-side variant in the ablation (all static shapes).
        compiled = torch.compile(matmul, dynamic=False)
        out = compiled(A, B)
        torch.testing.assert_close(out, torch.matmul(A, B), rtol=1e-4, atol=1e-4)
        compiled_stats = time_call(compiled, A, B, runs=runs)
        fmt(f"{name} torch.compile(inductor)", M, K, N, *compiled_stats)

        torch._dynamo.reset()  # avoid cache growth/guard buildup across shapes


if __name__ == "__main__":
    default_threads = torch.get_num_threads()
    run("single-threaded (matches every other ablation number)", num_threads=1)
    run(f"default thread count", num_threads=default_threads)
