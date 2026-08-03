# Google Colab Free GPU Trial

Use this notebook-style workflow to test the CUDA/NVPTX side of the project on
Google Colab's free GPU runtime. The goal is to find out whether Colab can link
and launch the MLIR-generated CUDA path before renting a dedicated GPU VM.

## Colab CLI Option

The Google Colab CLI can also run this workflow from your local terminal. This
is usually more convenient for this project because we need to upload archives,
run shell commands, inspect generated MLIR/LLVM/PTX artifacts, and possibly keep
an interactive shell open.

Install the CLI locally:

```bash
pip install google-colab-cli
```

Create a GPU runtime. Start with T4 on the free plan:

```bash
colab new -s mlir-gpu --gpu T4
colab status -s mlir-gpu
```

Upload the prepared bundles:

```bash
colab upload -s mlir-gpu colab_bundle/colab_compiler_stack.tar.gz /content/colab_compiler_stack.tar.gz
colab upload -s mlir-gpu colab_bundle/ML-compiler-exercise-source.tar.gz /content/ML-compiler-exercise-source.tar.gz
```

Open an interactive remote shell:

```bash
colab console -s mlir-gpu
```

Inside that remote shell, run the same setup commands from sections 3-8 below,
but use `/content/colab_compiler_stack.tar.gz` and
`/content/ML-compiler-exercise-source.tar.gz` instead of Google Drive paths:

```bash
sudo tar xzf /content/colab_compiler_stack.tar.gz -C /
mkdir -p /content/mlir_project
tar xzf /content/ML-compiler-exercise-source.tar.gz -C /content/mlir_project
```

When finished, download useful artifacts and stop the runtime:

```bash
colab download -s mlir-gpu /content/mlir_project/src/sample/gpu/sample_nvptx_isa.mlir ./sample_nvptx_isa.mlir
colab download -s mlir-gpu /content/mlir_project/src/sample/gpu/sample_gpu_dialect.mlir ./sample_gpu_dialect.mlir
colab log -s mlir-gpu -o colab_mlir_gpu_trial.ipynb
colab stop -s mlir-gpu
```

If the CLI cannot allocate a free GPU, fall back to the notebook workflow below
or try again later.

## 0. Prepare Uploads Locally

From the host that has the running Docker container:

```bash
cd /home/fuzzserver/dl-compiler-project/ML-compiler-exercise
bash tools/package_colab_bundle.sh
```

Upload these files to Google Drive:

```text
colab_bundle/colab_compiler_stack.tar.gz
colab_bundle/ML-compiler-exercise-source.tar.gz
```

The compiler stack is large because it includes the built LLVM/MLIR/torch-mlir
tree and Python virtual environment. This avoids rebuilding LLVM inside Colab.

## 1. Enable GPU In Colab

In Colab:

```text
Runtime -> Change runtime type -> Hardware accelerator -> GPU
```

Then run:

```bash
!nvidia-smi
!which ptxas || true
!nvcc --version || true
```

If `nvidia-smi` does not show an NVIDIA GPU, stop here and reconnect to a GPU
runtime.

## 2. Mount Drive And Unpack

```python
from google.colab import drive
drive.mount('/content/drive')
```

Adjust `BUNDLE_DIR` if you uploaded the files somewhere else:

```bash
%env BUNDLE_DIR=/content/drive/MyDrive/colab_bundle

!sudo tar xzf "$BUNDLE_DIR/colab_compiler_stack.tar.gz" -C /
!mkdir -p /content/mlir_project
!tar xzf "$BUNDLE_DIR/ML-compiler-exercise-source.tar.gz" -C /content/mlir_project
```

## 3. Set Environment

```bash
%env PATH=/opt/venv/bin:/build/build/bin:/usr/local/cuda/bin:/usr/bin:/bin
%env PYTHONPATH=/build/build/tools/mlir/python_packages/mlir_core:/build/build/tools/torch-mlir/python_packages/torch_mlir
%env LD_LIBRARY_PATH=/build/build/lib:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu/openblas-pthread
```

```bash
!mlir-opt --version
!torch-mlir-opt --version
!python3 - <<'PY'
import torch
import torch_mlir
print("torch", torch.__version__)
print("torch-mlir import OK")
PY
```

## 4. Install System Packages

```bash
!sudo apt-get update -y
!sudo apt-get install -y build-essential cmake ninja-build clang lld libopenblas-dev
```

## 5. Configure And Build `tutorial-opt`

```bash
%cd /content/mlir_project

!cmake -S . -B build-ninja -G Ninja \
  -DLLVM_DIR=/build/build/lib/cmake/llvm \
  -DMLIR_DIR=/build/build/lib/cmake/mlir \
  -DTORCH_MLIR_BUILD_DIR=/build/build \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_BUILD_TYPE=Release

!cmake --build build-ninja --target tutorial-opt -j2
!cmake --build build-ninja --target check-mlir-tutorial
```

## 6. Run CPU Sanity Check

```bash
%cd /content/mlir_project/src/sample
!python3 lower_sample_model.py
!bash run_tiled_mlir_pipeline.sh
!./tiled.out
```

This proves the PyTorch -> torch-mlir -> custom tiled MLIR -> LLVM -> native
CPU executable path still works in Colab.

## 7. Generate GPU/NVPTX Artifacts

```bash
%cd /content/mlir_project/src/sample/gpu
!cp ../sample_model_linalg.mlir .
!bash run_gpu_dialect_pipeline.sh
!bash run_mlir_pipeline.sh
!grep -n "gpu.binary\\|target sm_" sample_nvptx_isa.mlir | head
!ls -lh sample_gpu_dialect.mlir sample_nvptx_isa.mlir sample.ll sample.o
```

This proves the custom tiled MLIR can reach GPU dialect and NVPTX/PTX codegen.

## 8. Try Linking The GPU Runner

```bash
%env CUDA_HOME=/usr/local/cuda
%env MLIR_BUILD_DIR=/build/build
!bash compile.sh
```

If this succeeds:

```bash
!./a.out
```

If it fails, the most likely error is:

```text
MLIR CUDA runtime library not found
```

That means the imported MLIR build can generate NVPTX, but it does not include
the MLIR CUDA runtime library needed to launch kernels from a native executable.
At that point, the Colab trial has still been useful: it confirms GPU codegen
works, but launch/evaluation needs either a rebuilt CUDA-enabled MLIR runtime or
a rented GPU VM where we can control the full build.

## 9. What To Record

Record:

- GPU model from `nvidia-smi`.
- Whether `ptxas` exists.
- Whether `run_mlir_pipeline.sh` generates `sample_nvptx_isa.mlir`.
- Whether `compile.sh` links `a.out`.
- Whether `./a.out` launches and prints output.

If `./a.out` runs, the next step is adding benchmark scripts for tiled vs
untiled GPU MLIR.
