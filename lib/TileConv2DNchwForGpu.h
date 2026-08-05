#ifndef MLIR_TUTORIAL_TILE_CONV2D_NCHW_FOR_GPU_H
#define MLIR_TUTORIAL_TILE_CONV2D_NCHW_FOR_GPU_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"

namespace mlir::tutorial {

struct TileConv2DNchwForGpuPass
    : public PassWrapper<TileConv2DNchwForGpuPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TileConv2DNchwForGpuPass)

  TileConv2DNchwForGpuPass() = default;
  TileConv2DNchwForGpuPass(const TileConv2DNchwForGpuPass &other)
      : PassWrapper(other) {
    threads = other.threads;
  }

  StringRef getArgument() const final { return "tile-conv2d-nchw-for-gpu"; }

  StringRef getDescription() const final {
    return "Map bufferized f32 NCHW convolution to GPU block/thread loops";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, func::FuncDialect, linalg::LinalgDialect,
                memref::MemRefDialect, scf::SCFDialect>();
  }

  Option<int64_t> threads{
      *this, "threads",
      llvm::cl::desc("Linear output elements assigned to each GPU block"),
      llvm::cl::init(256)};

  void runOnOperation() final;
};

} // namespace mlir::tutorial

#endif // MLIR_TUTORIAL_TILE_CONV2D_NCHW_FOR_GPU_H
