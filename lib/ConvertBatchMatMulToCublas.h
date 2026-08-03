#ifndef MLIR_TUTORIAL_CONVERT_BATCH_MATMUL_TO_CUBLAS_H
#define MLIR_TUTORIAL_CONVERT_BATCH_MATMUL_TO_CUBLAS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"

namespace mlir::tutorial {

struct ConvertBatchMatMulToCublasPass
    : public PassWrapper<ConvertBatchMatMulToCublasPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertBatchMatMulToCublasPass)

  StringRef getArgument() const final {
    return "convert-batch-matmul-to-cublas";
  }

  StringRef getDescription() const final {
    return "Convert bufferized f32 batch matmul ops to a cuBLAS runtime call";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    linalg::LinalgDialect, memref::MemRefDialect>();
  }

  void runOnOperation() final;
};

} // namespace mlir::tutorial

#endif // MLIR_TUTORIAL_CONVERT_BATCH_MATMUL_TO_CUBLAS_H
