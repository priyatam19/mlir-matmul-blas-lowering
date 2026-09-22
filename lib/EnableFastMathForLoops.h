#ifndef MLIR_TUTORIAL_ENABLE_FAST_MATH_FOR_LOOPS_H
#define MLIR_TUTORIAL_ENABLE_FAST_MATH_FOR_LOOPS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::tutorial {

// LLVM's loop vectorizer refuses to reorder a floating-point reduction
// (e.g. the K-loop `arith.mulf`+`arith.addf` that `convert-linalg-to-loops`
// produces from `linalg.matmul`) unless the instructions are explicitly
// marked safe to reassociate. Without this, neither `-mcpu=native` nor a
// downstream `-ffast-math` compiler flag has any effect: `-ffast-math`
// only shapes code the frontend generates from source, it does not retrofit
// fastmath flags onto arith ops that already exist without them.
struct EnableFastMathForLoopsPass
    : public PassWrapper<EnableFastMathForLoopsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EnableFastMathForLoopsPass)

  StringRef getArgument() const final { return "enable-fastmath-for-loops"; }

  StringRef getDescription() const final {
    return "Set reassoc+contract fastmath flags on arith.mulf/arith.addf so "
           "LLVM's loop vectorizer may reorder generic-lowered reduction "
           "loops (e.g. the linalg-to-loops matmul fallback)";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect>();
  }

  void runOnOperation() final;
};

} // namespace mlir::tutorial

#endif // MLIR_TUTORIAL_ENABLE_FAST_MATH_FOR_LOOPS_H
