#ifndef MLIR_TUTORIAL_CONVERT_CONV2D_NCHW_TO_CUDNN_H
#define MLIR_TUTORIAL_CONVERT_CONV2D_NCHW_TO_CUDNN_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"

namespace mlir::tutorial {

struct ConvertConv2DNchwToCudnnPass
    : public PassWrapper<ConvertConv2DNchwToCudnnPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertConv2DNchwToCudnnPass)

  StringRef getArgument() const final { return "convert-conv2d-nchw-to-cudnn"; }

  StringRef getDescription() const final {
    return "Convert bufferized f32 NCHW/FCHW convolutions to a cuDNN call";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    linalg::LinalgDialect, memref::MemRefDialect>();
  }

  void runOnOperation() final;
};

} // namespace mlir::tutorial

#endif // MLIR_TUTORIAL_CONVERT_CONV2D_NCHW_TO_CUDNN_H
