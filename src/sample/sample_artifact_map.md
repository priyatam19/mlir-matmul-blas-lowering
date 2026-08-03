# `src/sample` Artifact Map

Date: 2026-07-07

This file tells you what each important artifact in `src/sample` is for and why it exists.

## High-Level Flow

```text
PyTorch model
  -> torch-mlir import
  -> linalg-level MLIR
  -> buffered MLIR
  -> LLVM dialect MLIR
  -> LLVM IR
  -> object file
  -> native executable called from C++
```

## Core Source Files

### `model.py`

Role:

- defines the original PyTorch model

Use it to answer:

- what computation is being compiled?
- what shapes and ops should appear later?

### `lower_sample_model.py`

Role:

- imports the PyTorch model into MLIR via `torch-mlir`

Expected output:

- `sample_model_linalg.mlir`

### `run_sample_model.py`

Role:

- runs the eager PyTorch model for correctness reference

### `benchmark_sample_model.py`

Role:

- measures baseline PyTorch inference cost for the toy workload

### `sample_call.cpp`

Role:

- native C++ caller for the compiled model
- demonstrates the ABI boundary between generated code and host code

## Baseline Pipeline Artifacts

### `sample_model_linalg.mlir`

Stage:

- first major MLIR artifact after import

What it represents:

- tensor-oriented computation in a higher-level dialect form
- this is where you should still clearly recognize model-level ops like matmul

### `sample_model_buf_linalg.mlir`

Stage:

- post-bufferization MLIR

What changed:

- tensor semantics start becoming memory/buffer oriented
- useful for understanding how high-level tensor code is prepared for lower-level codegen

### `sample_model_llvm.mlir`

Stage:

- MLIR lowered into LLVM dialect form

What changed:

- much closer to low-level execution and ABI concerns
- wrapper and calling convention details start becoming visible

### `sample_model_llvm_ir.ll`

Stage:

- translated LLVM IR text

What changed:

- this is outside MLIR proper
- now you are in conventional LLVM IR territory

### `sample_model_llvm_ir.o`

Stage:

- compiled object file from LLVM IR

### `a.out`

Stage:

- linked executable using generated object code and the C++ caller

## Tiled Pipeline Artifacts

### `sample_model_tiled_buf_linalg.mlir`

Stage:

- bufferized MLIR after custom tiling pass

Why it matters:

- first place to inspect whether the tiling transform changed structure meaningfully

### `sample_model_tiled_llvm.mlir`

Stage:

- tiled version lowered to LLVM dialect MLIR

### `sample_model_tiled_llvm_ir.ll`

Stage:

- tiled version translated to LLVM IR

### `sample_model_tiled_llvm_ir.o`

Stage:

- object file for the tiled path

### `tiled.out`

Stage:

- executable for the tiled path

## GPU / NVPTX Artifacts

These are conceptually useful even if you do not yet have a full CUDA/Triton environment.

### `gpu/sample_gpu_dialect.mlir`

Role:

- demonstrates GPU-target lowering in MLIR dialect form

### `gpu/sample_nvptx_isa.mlir`

Role:

- shows embedded PTX/NVPTX-targeted output

### `gpu/sample.ll`

Role:

- LLVM IR related to the GPU lowering path

### `gpu/sample.o`

Role:

- object artifact from the GPU path

## Pipeline Scripts

### `run_mlir_pipeline.sh`

Pipeline summary:

1. run custom tool for linalg-to-bufferization
2. lower buffered MLIR to LLVM dialect MLIR
3. translate MLIR to LLVM IR
4. compile LLVM IR to object file
5. link executable with C++ host code

### `run_tiled_mlir_pipeline.sh`

Pipeline summary:

1. apply custom tiling pass
2. bufferize
3. lower to LLVM dialect MLIR
4. translate to LLVM IR
5. compile and link tiled executable

## Custom Pass Connection

The key custom pass for `src/sample` is tied to:

- `../../lib/TileMatMulForCache.cpp`
- `../../lib/TileMatMulForCache.h`

This is the main bridge from “using a compiler pipeline” to “understanding and extending a compiler pipeline.”

## What To Compare Carefully

When studying this directory, compare these pairs:

1. `sample_model_linalg.mlir` vs `sample_model_buf_linalg.mlir`
2. `sample_model_buf_linalg.mlir` vs `sample_model_llvm.mlir`
3. `sample_model_llvm.mlir` vs `sample_model_llvm_ir.ll`
4. `sample_model_buf_linalg.mlir` vs `sample_model_tiled_buf_linalg.mlir`
5. `sample_model_llvm_ir.ll` vs `sample_model_tiled_llvm_ir.ll`

## What This Directory Teaches Well

- end-to-end lowering structure
- bufferization
- MLIR-to-LLVM transition
- native call boundary
- custom pass insertion
- tiling as a concrete transform
- NVPTX-target lowering concepts

## What This Directory Does Not Teach Well

- TorchInductor internals
- Triton kernel programming
- CUDA runtime benchmarking as a primary workflow
- real `torch.compile -> Triton` generated kernel analysis

Use this directory for compiler-pipeline understanding, then do the separate Triton/CUDA project for the main Track A resume project.
