#ifndef MLIR_TUTORIAL_LOWER_CONTRACTION_TO_GPU_H
#define MLIR_TUTORIAL_LOWER_CONTRACTION_TO_GPU_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::tutorial {

struct LowerContractionToGpuPass
    : public PassWrapper<LowerContractionToGpuPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerContractionToGpuPass)

  LowerContractionToGpuPass() = default;
  LowerContractionToGpuPass(const LowerContractionToGpuPass &other)
      : PassWrapper(other) {
    strategy = other.strategy;
    target = other.target;
    blockM = other.blockM;
    blockN = other.blockN;
    blockK = other.blockK;
    threads = other.threads;
    vectorWidth = other.vectorWidth;
    stages = other.stages;
  }

  StringRef getArgument() const final { return "lower-contraction-to-gpu"; }

  StringRef getDescription() const final {
    return "Lower bufferized contractions to shared-memory GPU kernels";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect, gpu::GPUDialect,
                    linalg::LinalgDialect, memref::MemRefDialect,
                    nvgpu::NVGPUDialect, scf::SCFDialect,
                    vector::VectorDialect>();
  }

  Option<std::string> strategy{
      *this, "strategy",
      llvm::cl::desc(
          "Kernel strategy: shared-fp32, tensorcore-tf32, or autotuned"),
      llvm::cl::init("shared-fp32")};
  Option<std::string> target{
      *this, "target", llvm::cl::desc("GPU architecture profile"),
      llvm::cl::init("sm_89")};
  Option<int64_t> blockM{*this, "block-m",
                         llvm::cl::desc("CTA output rows"),
                         llvm::cl::init(64)};
  Option<int64_t> blockN{*this, "block-n",
                         llvm::cl::desc("CTA output columns"),
                         llvm::cl::init(64)};
  Option<int64_t> blockK{*this, "block-k",
                         llvm::cl::desc("Reduction elements per K tile"),
                         llvm::cl::init(16)};
  Option<int64_t> threads{*this, "threads",
                          llvm::cl::desc("Threads per CTA"),
                          llvm::cl::init(256)};
  Option<int64_t> vectorWidth{
      *this, "vector-width",
      llvm::cl::desc("Contiguous elements per cooperative global load"),
      llvm::cl::init(4)};
  Option<int64_t> stages{
      *this, "stages", llvm::cl::desc("Workgroup-memory pipeline stages"),
      llvm::cl::init(1)};

  void runOnOperation() final;
};

} // namespace mlir::tutorial

#endif // MLIR_TUTORIAL_LOWER_CONTRACTION_TO_GPU_H
