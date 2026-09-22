# TorchDynamo Custom Backend

A custom `torch.compile` backend (`mlir_backend.py`) that lowers supported
subgraphs through this project's own MLIR pipeline instead of Inductor, and
falls back to eager execution for anything it doesn't recognize. This is the
extension `docs/triton_first_roadmap.md` named as a "next extension" and
`src/torch_compile/` previously had no content for.

## What's real here

Everything below actually runs and is numerically verified against eager --
this is not a sketch of the mechanism, it closes the loop:

1. **Dynamo capture.** `torch.compile(model, backend=mlir_tutorial_backend)`
   gets Dynamo to intercept `Sample.forward` (`src/sample/model.py`), trace it
   to an FX graph, and hand our backend the traced `GraphModule` plus one
   example tensor per graph placeholder.
2. **Accept/reject.** The backend walks the graph's nodes against a small
   allow-list (`operator.add`, `torch.nn.functional.linear`, the `clamp`
   method -- exactly what `Sample.forward` uses). Anything else -> the
   backend returns `gm.forward` unmodified, i.e. genuine eager fallback, with
   the specific unsupported node printed. This is the same kind of decision
   a Dynamo graph break makes, just made explicitly by this backend instead
   of by Dynamo's own guard machinery.
3. **Real lowering.** On accept, it calls
   `torch_mlir.fx.export_and_import(gm, *example_inputs, output_type=OutputType.LINALG_ON_TENSORS)`
   and runs the result through the exact passes `src/sample/run_mlir_pipeline.sh`
   uses: `tutorial-opt --linalg-to-bufferization`, then
   `tutorial-opt --llvm-request-c-wrappers --bufferization-to-llvm` (this is
   where this project's own `ConvertMatmulToBlasLibraryCallPass` fires and
   rewrites `linalg.matmul` into an OpenBLAS call -- it is genuinely
   exercised, not bypassed), `mlir-translate -mlir-to-llvmir`, `llc`, then
   links a shared library with `clang++`.
4. **Real execution.** The shared library is loaded with `ctypes` and called
   in-process through the same `_mlir_ciface_*` / `MemRefDescriptor` ABI
   `src/sample/sample_call.cpp` hand-writes in C++.

`demo.py` runs `Sample()` three ways -- eager, `torch.compile` with default
Inductor, and `torch.compile` with this backend -- and asserts all three
agree, then runs a second module using `torch.sin` (not in the allow-list) to
show the fallback path actually firing.

## A design choice worth calling out

Dynamo lifts closed-over `nn.Parameter`s (`self.param`, `self.linear.weight`,
`self.linear.bias`) to explicit graph placeholders before this backend ever
sees the graph. That means `torch_mlir.fx.export_and_import` keeps them as
**runtime memref arguments** in the exported MLIR, not baked-in
`dense_resource` constants -- unlike `src/sample/lower_sample_model.py`,
which exports the module directly and gets its weights folded into
constants at export time. Practically: this backend's compiled `.so` is
cached per input *shape* (see `_graph_key`), and serves any `Sample`
instance with that shape regardless of its actual weight values, which is
closer to how a real inference backend wants to cache compiled artifacts.

## A real bug worth recording

The first working version had correct shapes but wrong values on ~2/3 of
output elements. Root cause: the MLIR C-interface convention returns a
memref *by value* -- the callee allocates its own output buffer and
overwrites every field of the descriptor you pass in, including the data
pointer. Reading the result back from the buffer *you* pre-allocated (the
natural first instinct) reads stale memory; you have to read back through
`out_desc.aligned` *after* the call, once the callee has repointed it. Fixed
in `_call_compiled`.

## Running it

Requires `torch_mlir` and a `tutorial-opt` built against the same LLVM/MLIR
commit it was built against -- not pip-installable against a modern stable
PyTorch (torch-mlir's prebuilt wheels are pinned to a Jan-2024 nightly torch
+ cp311; see the project's own `docker/Dockerfile.cuda`, which builds it from
source). Use the existing `torch-mlir-dev` image:

```bash
docker run -d --name mlir-backend-dev \
  -v /path/to/ML-compiler-exercise:/workspace/project \
  -w /workspace/project torch-mlir-dev:latest sleep infinity

# Build tutorial-opt against the image's LLVM/MLIR tree (only needed once)
docker exec mlir-backend-dev cmake -G Ninja -B /workspace/project/build-docker \
  -S /workspace/project \
  -DLLVM_DIR=/build/build/lib/cmake/llvm \
  -DMLIR_DIR=/build/build/lib/cmake/mlir \
  -DTORCH_MLIR_BUILD_DIR=/build/build \
  -DBUILD_DEPS=ON -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Debug
docker exec mlir-backend-dev ninja -C /workspace/project/build-docker tutorial-opt

docker exec -e TUTORIAL_OPT=/workspace/project/build-docker/tools/tutorial-opt \
  mlir-backend-dev python3 src/torch_compile/demo.py
```

(`build-docker/` matches the repo's existing `build*/` gitignore pattern.)

## Known limitations (scope, not oversights)

- No partial-graph lowering: an unsupported node rejects the *whole* graph,
  matching Dynamo's per-graph (not per-node) compile unit here -- a real
  backend integration would want Dynamo's own graph-break splitting so only
  the unsupported slice falls back.
- The compiled-artifact cache lives in `/tmp` for the process lifetime only;
  nothing is persisted across runs the way `TutorialAutotuneRuntime`'s
  autotune cache is.
- `output_shape` is discovered by running the graph eagerly once per new
  cache key rather than reading it out of the exported MLIR's function
  signature -- correct, but avoidable overhead.
- Only 2-D/1-D contiguous float32 tensors are supported by the `ctypes`
  marshaling in `_as_memref`/`_call_compiled`.
