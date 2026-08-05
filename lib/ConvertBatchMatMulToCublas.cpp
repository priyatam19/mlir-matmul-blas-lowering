#include "lib/ConvertBatchMatMulToCublas.h"

#include "mlir/IR/PatternMatch.h"

using namespace mlir;

namespace mlir::tutorial {
namespace {

constexpr StringLiteral kRuntimeFunction =
    "tutorial_cublas_sgemm_strided_batched_f32";

bool hasSupportedType(Value value) {
  auto type = dyn_cast<MemRefType>(value.getType());
  if (!type || type.getRank() != 3 || !type.getElementType().isF32())
    return false;

  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(type.getStridesAndOffset(strides, offset)))
    return false;
  return ShapedType::isDynamic(strides.back()) || strides.back() == 1;
}

bool isSupportedBatchMatmul(linalg::BatchMatmulOp batchMatmul) {
  if (batchMatmul->getNumResults() != 0 ||
      !linalg::BatchMatmulOp::isDefaultIndexingMaps(
          batchMatmul.getIndexingMapsAttr()))
    return false;

  auto inputs = batchMatmul.getDpsInputs();
  auto outputs = batchMatmul.getDpsInits();
  return inputs.size() == 2 && outputs.size() == 1 &&
         hasSupportedType(inputs[0]) && hasSupportedType(inputs[1]) &&
         hasSupportedType(outputs[0]);
}

MemRefType getRuntimeMemRefType(MLIRContext *context) {
  constexpr int64_t dynamic = ShapedType::kDynamic;
  auto layout =
      StridedLayoutAttr::get(context, dynamic, {dynamic, dynamic, dynamic});
  return MemRefType::get({dynamic, dynamic, dynamic}, Float32Type::get(context),
                         layout);
}

func::FuncOp getOrCreateRuntimeFunction(ModuleOp module, OpBuilder &builder,
                                        MemRefType runtimeType) {
  if (auto function = module.lookupSymbol<func::FuncOp>(kRuntimeFunction))
    return function;

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  auto functionType =
      builder.getFunctionType({runtimeType, runtimeType, runtimeType}, {});
  auto function = func::FuncOp::create(builder, module.getLoc(),
                                       kRuntimeFunction, functionType);
  function.setPrivate();
  return function;
}

} // namespace

void ConvertBatchMatMulToCublasPass::runOnOperation() {
  ModuleOp module = getOperation();
  SmallVector<linalg::BatchMatmulOp> worklist;
  module.walk([&](linalg::BatchMatmulOp batchMatmul) {
    if (isSupportedBatchMatmul(batchMatmul))
      worklist.push_back(batchMatmul);
  });

  IRRewriter rewriter(&getContext());
  MemRefType runtimeType = getRuntimeMemRefType(&getContext());
  func::FuncOp runtimeFunction;
  for (linalg::BatchMatmulOp batchMatmul : worklist) {
    rewriter.setInsertionPoint(batchMatmul);
    if (!runtimeFunction)
      runtimeFunction =
          getOrCreateRuntimeFunction(module, rewriter, runtimeType);

    SmallVector<Value> operands;
    for (Value operand : llvm::concat<Value>(batchMatmul.getDpsInputs(),
                                             batchMatmul.getDpsInits())) {
      operands.push_back(memref::CastOp::create(rewriter, batchMatmul.getLoc(),
                                                runtimeType, operand));
    }
    func::CallOp::create(rewriter, batchMatmul.getLoc(), runtimeFunction,
                         operands);
    rewriter.eraseOp(batchMatmul);
  }
}

} // namespace mlir::tutorial
