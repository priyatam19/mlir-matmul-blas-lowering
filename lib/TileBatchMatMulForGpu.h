#ifndef MLIR_TUTORIAL_TILE_BATCH_MATMUL_FOR_GPU_H
#define MLIR_TUTORIAL_TILE_BATCH_MATMUL_FOR_GPU_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"

namespace mlir::tutorial {

struct TileBatchMatMulForGpuPass
    : public PassWrapper<TileBatchMatMulForGpuPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TileBatchMatMulForGpuPass)

  TileBatchMatMulForGpuPass() = default;
  TileBatchMatMulForGpuPass(const TileBatchMatMulForGpuPass &other)
      : PassWrapper(other) {
    blockM = other.blockM;
    blockN = other.blockN;
  }

  StringRef getArgument() const final { return "tile-batch-matmul-for-gpu"; }

  StringRef getDescription() const final {
    return "Map bufferized f32 batch matmul ops to GPU block/thread loops";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, func::FuncDialect, linalg::LinalgDialect,
                memref::MemRefDialect, scf::SCFDialect>();
  }

  Option<int64_t> blockM{
      *this, "block-m",
      llvm::cl::desc("Output rows computed by each GPU thread block"),
      llvm::cl::init(8)};
  Option<int64_t> blockN{
      *this, "block-n",
      llvm::cl::desc("Output columns computed by each GPU thread block"),
      llvm::cl::init(32)};

  void runOnOperation() final;
};

} // namespace mlir::tutorial

#endif // MLIR_TUTORIAL_TILE_BATCH_MATMUL_FOR_GPU_H
