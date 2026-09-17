# Triton-First Roadmap

Date: 2026-07-07

## Current Milestone

Add handwritten Triton kernels before integrating TorchDynamo or TorchInductor.
This keeps the GPU programming model visible:

```text
program ids -> block offsets -> masked loads -> tl.dot -> masked stores
```

Implemented starting points:

- `src/triton_kernels/vector_add.py`
- `src/triton_kernels/matmul.py`
- `src/triton_kernels/benchmarks/benchmark_shapes.py`
- `src/triton_kernels/fused_linear_clamp.py` -- fused kernel from Near-Term
  Extension 1 below, now implemented
- `src/triton_kernels/fused_linear_gelu.py` -- same idea at the `gpt2_ffn`
  shape, with a bias and GELU added
- `src/torch_compile/` -- the TorchDynamo/TorchInductor inspection path from
  Near-Term Extension 2 below, now implemented

## Study Order

1. Run `vector_add.py` to verify PyTorch, Triton, CUDA, and GPU visibility.
2. Run `matmul.py` with the default shape and inspect the kernel structure.
3. Run `benchmark_shapes.py` to compare against the repo's MLIR benchmark shapes.
4. Tune `--block-m`, `--block-n`, and `--block-k`.
5. Record results next to `docs/benchmark_results.md`.

## Near-Term Extensions

1. ~~Add a fused linear activation kernel: `C = clamp(A @ B + bias, 0, 1)`,
   mirroring `src/sample/model.py`.~~ **Done** --
   `src/triton_kernels/fused_linear_clamp.py`. Also added
   `fused_linear_gelu.py` for the `gpt2_ffn` shape, not originally scoped
   here but the same idea and directly useful for the tensor-parallel study
   in `docs/tensor_parallel_placement_analysis.md`.

2. ~~Add a TorchDynamo/TorchInductor inspection path.~~ **Done** --
   `src/torch_compile/` registers a custom Dynamo backend
   (`mlir_backend.py`) that lowers supported subgraphs through this
   project's own MLIR pipeline instead of Inductor, verified end-to-end
   inside the `torch-mlir-dev` container.

3. Compare handwritten Triton kernels with generated Inductor kernels.
   **Partially done**: `src/torch_compile/demo.py` runs eager, Inductor,
   the MLIR backend, and `fused_linear_clamp` side by side on the same
   model -- but no environment in this repo currently has both `torch_mlir`
   and `triton`+CUDA together, so the Triton leg degrades to a labeled skip
   there rather than producing a real number. Needs a GPU session with both
   installed (see `src/torch_compile/README.md`).

4. Add an optional dependency file once the target CUDA/PyTorch version is known.
   Still open.

## Notes

The current host Python environment did not originally have `torch` or
`triton` installed; the fused kernels changed this in one specific way: both
are correctness-checked without a GPU via Triton's CPU interpreter
(`TRITON_INTERPRET=1 python3 <kernel>.py --interpret`), which needed `torch`
and `triton` installed in a local CPU-only venv (`dev/`, gitignored) to
verify. Real GPU timing for any script in this directory still requires an
actual CUDA environment (RunPod/Colab, as elsewhere in this project) and
still does not touch the repo's `requirements.txt`, which remains focused on
MLIR setup.
