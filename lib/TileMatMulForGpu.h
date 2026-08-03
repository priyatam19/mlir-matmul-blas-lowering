#ifndef MLIR_TUTORIAL_TILE_MATMUL_FOR_GPU_H
#define MLIR_TUTORIAL_TILE_MATMUL_FOR_GPU_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"

namespace mlir::tutorial {

struct TileMatMulForGpuPass
    : public PassWrapper<TileMatMulForGpuPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TileMatMulForGpuPass)

  TileMatMulForGpuPass() = default;
  TileMatMulForGpuPass(const TileMatMulForGpuPass &other) : PassWrapper(other) {
    blockM = other.blockM;
    blockN = other.blockN;
  }

  StringRef getArgument() const final { return "tile-matmul-for-gpu"; }

  StringRef getDescription() const final {
    return "Map bufferized f32 matmul ops to nested GPU block/thread loops";
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

#endif // MLIR_TUTORIAL_TILE_MATMUL_FOR_GPU_H
