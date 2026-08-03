# `src/sample` Pipeline Notes Template

Date:

Environment:

- Host:
- Container/image:
- Python version:
- Tool versions:
  - `torch-mlir-opt`:
  - `mlir-translate`:
  - `llc`:

## 1. Source Model

Files:

- `model.py`
- `run_sample_model.py`

Model summary:

- Input shape(s):
- Output shape(s):
- Main ops:
- Expected computation flow:

Notes:

-
-
-

## 2. Eager PyTorch Run

Command:

```bash
python3 run_sample_model.py
```

Observed output:

-
-

Correctness notes:

-
-

## 3. Imported MLIR

File:

- `sample_model_linalg.mlir`

Abstraction level:

-

Dominant dialects / ops:

-
-
-

What is easy to recognize:

-
-

What is still unclear:

-
-

## 4. Buffered MLIR

File:

- `sample_model_buf_linalg.mlir`

What changed from previous stage:

-
-
-

Where memory or buffer semantics appear:

-
-

Questions this stage answered:

-
-

## 5. LLVM Dialect MLIR

File:

- `sample_model_llvm.mlir`

What changed from previous stage:

-
-
-

Where low-level details become visible:

-
-

ABI / wrapper observations:

-
-

## 6. LLVM IR

File:

- `sample_model_llvm_ir.ll`

What changed from previous stage:

-
-
-

What looks familiar from classical compiler work:

-
-

What is MLIR-specific context that led here:

-
-

## 7. Native Object / Executable

Files:

- `sample_model_llvm_ir.o`
- `a.out`

Native call path summary:

-
-
-

C++ boundary observations from `sample_call.cpp`:

-
-
-

## 8. Baseline CPU Pipeline Summary

Command:

```bash
bash run_mlir_pipeline.sh
```

Pipeline stages in my own words:

1.
2.
3.
4.
5.

Biggest takeaways:

-
-
-

## 9. Benchmark Notes

Command(s):

```bash
python3 benchmark_sample_model.py
```

Results:

- Eager PyTorch:
- Native / MLIR path:

Cautions about interpretation:

-
-
-

## 10. Tiled Pipeline

Files:

- `sample_model_tiled_buf_linalg.mlir`
- `sample_model_tiled_llvm.mlir`
- `sample_model_tiled_llvm_ir.ll`
- `tiled.out`

Command:

```bash
bash run_tiled_mlir_pipeline.sh
```

Tiling configuration:

- `tile-m=`
- `tile-n=`
- `tile-k=`

Structural differences vs baseline:

-
-
-

Performance differences vs baseline:

-
-

## 11. Custom Pass Notes

Files:

- `../../lib/TileMatMulForCache.cpp`
- `../../lib/TileMatMulForCache.h`

Pass purpose in my own words:

-
-
-

How the pass is registered / invoked:

-
-

What operation(s) it transforms:

-
-

Why this is a transform pass:

-
-

## 12. Optional GPU / NVPTX Notes

Files:

- `gpu/sample_gpu_dialect.mlir`
- `gpu/sample_nvptx_isa.mlir`

Commands run:

```bash
cd gpu
# fill in what you ran
```

What this demonstrates:

-
-
-

What it does not demonstrate:

-
-

## 13. Final Understanding

End-to-end pipeline in 5-8 lines:

-
-
-
-
-

What this repo teaches me that is relevant to AI compiler roles:

-
-
-

What it does not cover and I still need from Track A:

-
-
-
