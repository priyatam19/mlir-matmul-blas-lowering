# Triton Kernel Track

This directory starts the Triton-first extension of the project. It is separate
from the existing `torch-mlir` pipeline on purpose:

```text
Manual Triton path:
PyTorch tensors -> Triton JIT kernel -> CUDA execution

Existing MLIR path:
PyTorch model -> torch-mlir -> linalg/bufferization/LLVM -> native executable
```

The first goal is to learn and benchmark handwritten Triton kernels before
connecting them to TorchDynamo or TorchInductor.

## Environment

These scripts require:

- PyTorch
- Triton
- CUDA-enabled PyTorch
- an NVIDIA GPU visible to the Python process

Basic install example:

```bash
python3 -m pip install torch triton
```

If you need a specific CUDA wheel, use the selector at:

```text
https://pytorch.org/get-started/locally/
```

The scripts check for missing packages and unavailable CUDA, then print setup
guidance instead of failing with an unhelpful import traceback.

## Smoke Test

Run the smallest kernel first:

```bash
cd src/triton_kernels
python3 vector_add.py
```

Expected result:

```text
vector_add: ok
```

## Single Matmul Benchmark

Run the default matrix multiply benchmark:

```bash
cd src/triton_kernels
python3 matmul.py
```

Try the shape from the larger MLIR benchmark:

```bash
python3 matmul.py --m 512 --k 256 --n 512
```

The script validates correctness against `torch.matmul` and prints timing for:

- handwritten Triton matmul
- `torch.matmul`

## Repo Benchmark Shapes

Run the same GEMM-like shapes used by `src/benchmarks/pytorch_bench.py`:

```bash
cd src/triton_kernels
python3 benchmarks/benchmark_shapes.py
```

Run one benchmark by name:

```bash
python3 benchmarks/benchmark_shapes.py --filter large_gemm
```

## Fused Kernels

Two hand-written fused kernels, each mirroring a real shape already used
elsewhere in this repo, each `@triton.autotune`-d over `BLOCK_M`/`BLOCK_N`/
`BLOCK_K`/`num_warps`:

- `fused_linear_clamp.py` -- `clamp((x + param) @ W.T + b, 0, 1)`, the exact
  computation in `src/sample/model.py`'s `Sample.forward` and the exact
  subgraph `src/torch_compile/mlir_backend.py` lowers through MLIR instead.
  Fuses the elementwise add, the GEMM, the bias add, and the clamp into one
  kernel launch.
- `fused_linear_gelu.py` -- `gelu(x @ W + b)` at this repo's own
  `gpt2_ffn.mlir` shape (`M=128, K=256, N=512`), extended with a bias and
  GELU to match the actual fusable unit of a transformer FFN's
  up-projection -- the op `docs/tensor_parallel_placement_analysis.md`
  column-shards across GPUs.

Both take a GPU to benchmark for real, same as everything else in this
directory. Unlike everything else in this directory, both can also be
**correctness-checked without one**, via Triton's CPU interpreter:

```bash
TRITON_INTERPRET=1 python3 fused_linear_clamp.py --interpret
TRITON_INTERPRET=1 python3 fused_linear_gelu.py --interpret
```

`--interpret` bypasses `@triton.autotune` (its benchmarking harness calls
into the active GPU driver even when kernel *execution* is running under the
interpreter) and launches the raw kernel with one fixed config instead --
enough to catch a logic bug, not to reproduce autotuned timing.

`fused_linear_clamp` is also wired into
`src/torch_compile/demo.py` as a fourth point of comparison alongside eager,
default Inductor, and this project's own MLIR backend, for the exact same
`Sample` model and weights. It degrades to a labeled skip there when triton
or CUDA aren't both available in the running environment (true of both
environments in this repo today -- see that file's docstring).

## What To Compare Next

Use these outputs next to the existing MLIR benchmark artifacts:

- `src/benchmarks/pytorch_bench.py`
- `src/benchmarks/*_notiled.out`
- `src/benchmarks/*_tiled.out`

This gives the first comparison table:

```text
PyTorch eager CPU
MLIR/OpenBLAS CPU
MLIR tiled CPU
PyTorch CUDA
Manual Triton CUDA
Fused Triton CUDA (fused_linear_clamp / fused_linear_gelu)
```

A TorchDynamo/TorchInductor inspection path now exists at
`src/torch_compile/` -- run its `demo.py` to compare generated Inductor
kernels, this project's own MLIR backend, and these handwritten Triton
kernels against each other for the same model.
