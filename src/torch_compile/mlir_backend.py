#!/usr/bin/env python3
"""A custom TorchDynamo backend that lowers supported subgraphs through this
project's own MLIR pipeline (torch_mlir -> tutorial-opt -> mlir-translate ->
llc -> a JIT-loaded shared library), and falls back to eager execution for
anything it doesn't recognize -- the same shape of decision a graph break
makes inside TorchDynamo itself, just made explicitly by this backend.

Mechanism
---------
`torch.compile(model, backend=mlir_tutorial_backend)` hands this function a
traced `torch.fx.GraphModule` plus one example input per placeholder. Dynamo
has already done the hard part (bytecode interception, guard installation);
this backend only has to decide whether it can compile the traced graph, and
if so, produce a callable that returns equivalent results.

Supported subgraph (`add` -> `linear` -> `clamp`, i.e. `src/sample/model.py`):
  1. Re-export the exact GraphModule Dynamo captured via
     `torch_mlir.fx.export_and_import(...)`. Because Dynamo lifts closed-over
     `nn.Parameter`s to explicit graph placeholders, the exported MLIR keeps
     them as *runtime* memref arguments rather than baked-in constants --
     unlike this repo's existing `lower_sample_model.py`, which exports the
     module directly and gets its weights folded into `dense_resource`
     constants. That means this backend's compiled artifact is cached per
     input *shape*, not per weight values: the same compiled kernel serves
     any `Sample` instance with the same input shape, which is closer to how
     a real inference backend wants to cache compiled code.
  2. Run it through the exact passes `run_mlir_pipeline.sh` uses:
     `--linalg-to-bufferization`, then `--llvm-request-c-wrappers
     --bufferization-to-llvm` (the latter is where this project's own
     `ConvertMatmulToBlasLibraryCallPass` fires and rewrites `linalg.matmul`
     into an OpenBLAS call), `mlir-translate -mlir-to-llvmir`, `llc`.
  3. Link the result into a shared library and call it in-process via
     `ctypes`, using the `MemRefDescriptor` C-interface layout
     `src/sample/sample_call.cpp` already hand-writes in C++.

Unsupported subgraph: return `gm.forward` unmodified -- eager execution,
with the specific unsupported node named on stderr so it's clear *why* this
backend declined it (the same information a Dynamo graph break would need to
report).

Known limitations (intentional scope for a first cut, not oversights):
  - Only the exact op set below is recognized; anything else -> eager
    fallback in full, no partial-graph lowering.
  - Compiled artifacts are cached by graph structure + input shapes for the
    lifetime of the process (see `_CACHE_DIR`); nothing is persisted to disk
    across runs the way `TutorialAutotuneRuntime`'s cache is.
  - Output shape is currently discovered by running the graph eagerly once
    per new cache key -- correct, but wasteful; a production version would
    read the shape straight out of the exported MLIR's function signature.
"""

from __future__ import annotations

import ctypes
import hashlib
import operator
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]

_SUPPORTED_CALL_FUNCTIONS = {operator.add, torch.nn.functional.linear}
_SUPPORTED_CALL_METHODS = {"clamp"}

_CACHE_DIR = Path(tempfile.gettempdir()) / "mlir_tutorial_dynamo_cache"
_CACHE_DIR.mkdir(exist_ok=True)


def _find_tutorial_opt() -> str | None:
    if override := os.environ.get("TUTORIAL_OPT"):
        return override
    for candidate in ("build-docker", "build-ninja"):
        path = _REPO_ROOT / candidate / "tools" / "tutorial-opt"
        if path.exists():
            return str(path)
    return None


def _is_supported(gm: torch.fx.GraphModule) -> bool:
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target not in _SUPPORTED_CALL_FUNCTIONS:
            print(f"[mlir_backend] unsupported call_function {node.target!r} -> eager fallback")
            return False
        if node.op == "call_method" and node.target not in _SUPPORTED_CALL_METHODS:
            print(f"[mlir_backend] unsupported call_method {node.target!r} -> eager fallback")
            return False
        if node.op == "call_module":
            print(f"[mlir_backend] unsupported call_module {node.target!r} -> eager fallback")
            return False
    return True


def _graph_key(gm: torch.fx.GraphModule, example_inputs) -> str:
    shapes = "-".join("x".join(map(str, t.shape)) for t in example_inputs)
    digest = hashlib.sha1((str(gm.graph) + shapes).encode()).hexdigest()[:16]
    return digest


def _run(cmd: list[str], stdout_path: Path | None = None) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"`{' '.join(cmd)}` failed:\n{result.stderr}")
    if stdout_path is not None:
        stdout_path.write_text(result.stdout)


def _compile_to_shared_object(gm: torch.fx.GraphModule, example_inputs, key: str, func_name: str) -> Path:
    so_path = _CACHE_DIR / f"{key}.so"
    if so_path.exists():
        return so_path

    tutorial_opt = _find_tutorial_opt()
    if tutorial_opt is None:
        raise RuntimeError("tutorial-opt not found (set TUTORIAL_OPT or build build-docker/build-ninja)")

    from torch_mlir import fx as torch_mlir_fx
    from torch_mlir.compiler_utils import OutputType

    workdir = _CACHE_DIR / key
    workdir.mkdir(exist_ok=True)

    module = torch_mlir_fx.export_and_import(
        gm, *example_inputs, output_type=OutputType.LINALG_ON_TENSORS, func_name=func_name
    )
    (workdir / "linalg.mlir").write_text(str(module))

    _run([tutorial_opt, "--linalg-to-bufferization", str(workdir / "linalg.mlir")],
         workdir / "bufferized.mlir")
    _run([tutorial_opt, "--llvm-request-c-wrappers", "--bufferization-to-llvm",
          str(workdir / "bufferized.mlir")], workdir / "llvm.mlir")
    _run(["mlir-translate", "-mlir-to-llvmir", str(workdir / "llvm.mlir")], workdir / "out.ll")
    _run(["llc", "--filetype=obj", "--relocation-model=pic", str(workdir / "out.ll"),
          "-o", str(workdir / "out.o")])
    _run(["clang++", "-shared", "-fPIC", str(workdir / "out.o"), "-o", str(so_path),
          "-lopenblas", "-lm"])
    return so_path


def _memref_type(rank: int) -> type[ctypes.Structure]:
    class MemRef(ctypes.Structure):
        _fields_ = [
            ("allocated", ctypes.POINTER(ctypes.c_float)),
            ("aligned", ctypes.POINTER(ctypes.c_float)),
            ("offset", ctypes.c_int64),
            ("sizes", ctypes.c_int64 * rank),
            ("strides", ctypes.c_int64 * rank),
        ]

    return MemRef


def _row_major_strides(shape: tuple[int, ...]) -> list[int]:
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return strides


def _as_memref(tensor: torch.Tensor):
    contig = tensor.detach().contiguous().to(torch.float32)
    buf = contig.numpy()
    ptr = buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    strides = _row_major_strides(tuple(buf.shape))
    desc = _memref_type(buf.ndim)(ptr, ptr, 0, tuple(buf.shape), tuple(strides))
    return desc, buf  # caller must keep `buf` alive as long as `desc` is used


def _call_compiled(so_path: Path, func_name: str, args: tuple[torch.Tensor, ...], output_shape) -> torch.Tensor:
    lib = ctypes.CDLL(str(so_path))
    fn = getattr(lib, f"_mlir_ciface_{func_name}")

    keepalive = []
    arg_descs = []
    for t in args:
        desc, buf = _as_memref(t)
        keepalive.append(buf)
        arg_descs.append(desc)

    # The callee returns its result memref by value over the C interface: it
    # allocates its own output buffer and overwrites every field of the
    # descriptor we pass in -- including `aligned`, which ends up pointing at
    # the callee's allocation, not anything we own. So the result has to be
    # read back through the (now-updated) descriptor, not through a buffer
    # we pre-allocated on our side.
    out_desc = _memref_type(len(output_shape))()
    fn(ctypes.byref(out_desc), *(ctypes.byref(d) for d in arg_descs))

    result = np.ctypeslib.as_array(out_desc.aligned, shape=tuple(out_desc.sizes)).copy()
    return torch.from_numpy(result)


def mlir_tutorial_backend(gm: torch.fx.GraphModule, example_inputs) -> Callable:
    """Custom Dynamo backend: lower to this project's MLIR pipeline, or fall
    back to eager. Pass directly as `torch.compile(model, backend=...)`."""

    if not _is_supported(gm):
        return gm.forward

    key = _graph_key(gm, example_inputs)
    func_name = f"fn_{key}"

    try:
        so_path = _compile_to_shared_object(gm, example_inputs, key, func_name)
    except Exception as exc:  # noqa: BLE001 - deliberate: any failure degrades to eager
        print(f"[mlir_backend] compilation failed ({exc}); eager fallback")
        return gm.forward

    with torch.no_grad():
        result = gm(*example_inputs)
        result = result[0] if isinstance(result, (tuple, list)) else result
        output_shape = tuple(result.shape)

    def compiled(*args: torch.Tensor):
        # Dynamo expects a return structure matching the FX graph's output
        # node (here a 1-tuple, `((clamp,),)` in the captured graph) -- a
        # bare tensor gets silently mis-flattened by the caller instead of
        # raising, so this is easy to get wrong.
        return (_call_compiled(so_path, func_name, args, output_shape),)

    return compiled


try:
    torch._dynamo.register_backend(name="mlir_tutorial")(mlir_tutorial_backend)
except Exception:
    pass  # registration is a convenience; callers can always pass the callable directly
