# Nsight Systems Summary

Nsight Systems 2024.6.2 traced one correctness-only invocation of ten
representative binaries after all formal timing and tuning runs. Collection
used CUDA, cuBLAS, cuDNN, NVTX, and OS runtime tracing with CPU sampling and
context-switch collection disabled. These diagnostic one-shot times include
profiler and lazy runtime initialization; the CUDA-event values in
`summary.csv` remain the formal steady-state measurements.

## Isolated operators

| Operator mode | Implementation kernel time | Grid | Block |
|---|---:|---:|---:|
| Long BMM untiled | 21.935284 ms | `512x512x12` | `1x1x1` |
| Long BMM block-thread | 0.310109 ms | `16x64x12` | `32x8x1` |
| Long BMM cuBLAS | 0.038592 ms/call | `4x4x12` | `256x1x1` |
| ResNet convolution untiled | 10.216179 ms | `56x64x1` | `1x1x1` |
| ResNet convolution block-thread | 0.263933 ms | `784x1x1` | `256x1x1` |
| ResNet convolution cuDNN | 0.015647 ms/call | two kernels | mixed |

The timeline confirms one implementation launch in every isolated custom case.
Changing the launch topology improves traced kernel time by 70.73x for long
BMM and 38.71x for ResNet convolution. The custom kernels remain 8.0x and
16.9x slower than their vendor-library kernels, respectively.

## Mixed programs

| Program mode | GPU kernels | Operator GPU time | Total GPU kernel time | Operator share | Stream-sync time |
|---|---:|---:|---:|---:|---:|
| Attention block-thread | 14 | 0.066848 ms | 2.321992 ms | 2.88% | 2.370997 ms |
| Attention vendor | 14 | 0.029664 ms | 2.232488 ms | 1.33% | 2.240860 ms |
| Residual block-thread | 9 | 0.031424 ms | 2.601220 ms | 1.21% | 2.628693 ms |
| Residual vendor | 11 | 0.016352 ms | 2.026891 ms | 0.81% | 2.036501 ms |

The attention block has two mapped BMM kernels, while the residual block has
two mapped convolution kernels. Nearly all remaining GPU time is spent in
generic elementwise, transpose, reduction, and normalization kernels launched
with one-thread blocks. Stream synchronization time closely tracks total GPU
kernel time, showing that this execution path serializes nearly every generic
kernel instead of overlapping host work or device execution.

## CPU and API observations

- CUDA launch API work is small relative to the long untiled kernels. Their
  bottleneck is inefficient GPU execution, not host launch latency.
- Mixed-program launch API totals are roughly 0.10-0.17 ms, but per-kernel
  stream synchronization serializes the graph and exposes all kernel latency.
- First-use managed allocation, CUDA module loading, cuBLAS initialization, and
  cuDNN lazy setup dominate one-shot process wall time. These costs are outside
  the preallocated steady-state CUDA-event interval, but a persistent runtime
  is required to amortize them in deployment.
- OS runtime `poll` and `ioctl` totals include CUDA driver and profiler service
  activity. They are useful for timeline correlation, not as application-only
  CPU execution measurements.

The next whole-model performance work should therefore target fusion and
proper GPU mapping of generic elementwise/reduction kernels. Shared-memory
tiling remains the next isolated BMM/convolution optimization for narrowing the
remaining gap to cuBLAS and cuDNN.

The `nsys/` directory contains the exported CUDA API, GPU-kernel, kernel-launch,
GPU-memory, and OS-runtime CSV reports. The interactive `.nsys-rep` timelines
and SQLite exports are retained in the complete local archive.
