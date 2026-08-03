#!/usr/bin/env python3
"""PyTorch single-threaded baseline for each benchmark shape."""
import torch, time, statistics

torch.set_num_threads(1)  # match MLIR single-thread mode

configs = [
    ("bert_attn",   128,  64, 128, 500),
    ("resnet_conv", 256, 128, 256, 200),
    ("gpt2_ffn",    128, 256, 512, 300),
    ("large_gemm",  512, 256, 512, 100),
]

for name, M, K, N, runs in configs:
    A = torch.randn(M, K)
    B = torch.randn(K, N)
    for _ in range(20):
        _ = torch.matmul(A, B)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter_ns()
        _ = torch.matmul(A, B)
        t1 = time.perf_counter_ns()
        times.append(t1 - t0)
    times.sort()
    avg = int(statistics.mean(times))
    p50 = times[len(times) // 2]
    p10 = times[len(times) // 10]
    p90 = times[len(times) * 9 // 10]
    print(f"  {name:<12} M={M} K={K} N={N}: avg={avg} ns  p50={p50}  p10={p10}  p90={p90}")
