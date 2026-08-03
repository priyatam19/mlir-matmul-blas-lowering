- `python lower_sample_model.py` to import the Pytorch model to mlir using torch-mlir
- `python run_sample_model.py` to print the output with sample input
- `python benchmark_sample_model.py` to benchmark the sample model
- `bash run_mlir_pipeline.sh` to build the CPU/OpenBLAS executable from the MLIR-lowered PyTorch model.
- `bash run_tiled_mlir_pipeline.sh` to run the custom `tile-matmul-for-cache` pass before lowering to LLVM.
- `cd gpu && bash run_gpu_dialect_pipeline.sh` to verify GPU dialect lowering with the custom tiling pass on machines without CUDA.
- `cd gpu && bash run_mlir_pipeline.sh` to run the CUDA/NVPTX lowering pipeline with the custom tiling pass and emit embedded PTX ISA in `sample_nvptx_isa.mlir`.
- `cd gpu && bash compile.sh` to link a GPU launch executable on a machine with CUDA driver/runtime libraries and MLIR CUDA runtime support.
- See `../../docs/CustomPassExtension.md` for the full extension workflow and verification commands.

### Benchnmark results:
- PyTorch avg. inference time (CPU): 0.000023 sec
- MLIR pipeline avg. inference time (CPU): 0.00000068546 sec
- MLIR pipeline avg. inference time (GPU): 0.001746 sec
