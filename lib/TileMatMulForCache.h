#ifndef MLIR_TUTORIAL_TILE_MATMUL_FOR_CACHE_H
#define MLIR_TUTORIAL_TILE_MATMUL_FOR_CACHE_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"

namespace mlir::tutorial {

struct TileMatMulForCachePass
    : public PassWrapper<TileMatMulForCachePass,
                         OperationPass<mlir::ModuleOp>> {
  TileMatMulForCachePass() = default;
  TileMatMulForCachePass(const TileMatMulForCachePass &other)
      : PassWrapper(other) {
    tileM = other.tileM;
    tileN = other.tileN;
    tileK = other.tileK;
  }

  StringRef getArgument() const final { return "tile-matmul-for-cache"; }

  StringRef getDescription() const final {
    return "Tile tensor-level linalg.matmul ops into scf.for loop nests";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect,
                    tensor::TensorDialect, func::FuncDialect>();
  }

  Option<int64_t> tileM{*this, "tile-m",
                        llvm::cl::desc("Tile size for the M dimension"),
                        llvm::cl::init(16)};
  Option<int64_t> tileN{*this, "tile-n",
                        llvm::cl::desc("Tile size for the N dimension"),
                        llvm::cl::init(16)};
  Option<int64_t> tileK{*this, "tile-k",
                        llvm::cl::desc("Tile size for the K dimension"),
                        llvm::cl::init(8)};

  void runOnOperation() final;
};

} // namespace mlir::tutorial

#endif // MLIR_TUTORIAL_TILE_MATMUL_FOR_CACHE_H
