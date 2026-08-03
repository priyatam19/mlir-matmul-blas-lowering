#include "lib/ConvertConv2DNchwToCudnn.h"

#include "mlir/IR/PatternMatch.h"

#include <array>

using namespace mlir;

namespace mlir::tutorial {
namespace {

constexpr StringLiteral kRuntimeFunction = "tutorial_cudnn_conv2d_nchw_f32";

FailureOr<std::array<int64_t, 2>> getPositivePair(DenseIntElementsAttr attr) {
  if (!attr || attr.getNumElements() != 2)
    return failure();
  std::array<int64_t, 2> values;
  size_t index = 0;
  for (APInt value : attr.getValues<APInt>())
    values[index++] = value.getSExtValue();
  if (values[0] <= 0 || values[1] <= 0)
    return failure();
  return values;
}

bool hasSupportedType(Value value, bool requireContiguous) {
  auto type = dyn_cast<MemRefType>(value.getType());
  if (!type || type.getRank() != 4 || !type.getElementType().isF32())
    return false;

  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(type.getStridesAndOffset(strides, offset)))
    return false;
  if (!ShapedType::isDynamic(strides.back()) && strides.back() != 1)
    return false;
  if (!requireContiguous)
    return true;

  int64_t expected = 1;
  for (int64_t dim = 3; dim >= 0; --dim) {
    if (!ShapedType::isDynamic(strides[dim]) && strides[dim] != expected)
      return false;
    int64_t size = type.getDimSize(dim);
    if (ShapedType::isDynamic(size) || ShapedType::isDynamic(expected))
      expected = ShapedType::kDynamic;
    else
      expected *= size;
  }
  return true;
}

bool isSupportedConv(linalg::Conv2DNchwFchwOp conv) {
  if (conv->getNumResults() != 0 ||
      failed(getPositivePair(conv.getStrides())) ||
      failed(getPositivePair(conv.getDilations())))
    return false;
  auto inputs = conv.getDpsInputs();
  auto outputs = conv.getDpsInits();
  return inputs.size() == 2 && outputs.size() == 1 &&
         hasSupportedType(inputs[0], false) &&
         hasSupportedType(inputs[1], true) &&
         hasSupportedType(outputs[0], false);
}

MemRefType getRuntimeMemRefType(MLIRContext *context) {
  constexpr int64_t dynamic = ShapedType::kDynamic;
  auto layout = StridedLayoutAttr::get(context, dynamic,
                                       {dynamic, dynamic, dynamic, dynamic});
  return MemRefType::get({dynamic, dynamic, dynamic, dynamic},
                         Float32Type::get(context), layout);
}

func::FuncOp getOrCreateRuntimeFunction(ModuleOp module, OpBuilder &builder,
                                        MemRefType runtimeType) {
  if (auto function = module.lookupSymbol<func::FuncOp>(kRuntimeFunction))
    return function;

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  auto indexType = builder.getIndexType();
  auto functionType =
      builder.getFunctionType({runtimeType, runtimeType, runtimeType, indexType,
                               indexType, indexType, indexType},
                              {});
  auto function = func::FuncOp::create(builder, module.getLoc(),
                                       kRuntimeFunction, functionType);
  function.setPrivate();
  return function;
}

} // namespace

void ConvertConv2DNchwToCudnnPass::runOnOperation() {
  ModuleOp module = getOperation();
  SmallVector<linalg::Conv2DNchwFchwOp> worklist;
  module.walk([&](linalg::Conv2DNchwFchwOp conv) {
    if (isSupportedConv(conv))
      worklist.push_back(conv);
  });

  IRRewriter rewriter(&getContext());
  MemRefType runtimeType = getRuntimeMemRefType(&getContext());
  func::FuncOp runtimeFunction;
  for (linalg::Conv2DNchwFchwOp conv : worklist) {
    rewriter.setInsertionPoint(conv);
    if (!runtimeFunction)
      runtimeFunction =
          getOrCreateRuntimeFunction(module, rewriter, runtimeType);

    SmallVector<Value> operands;
    for (Value operand :
         llvm::concat<Value>(conv.getDpsInputs(), conv.getDpsInits())) {
      operands.push_back(memref::CastOp::create(rewriter, conv.getLoc(),
                                                runtimeType, operand));
    }
    auto strides = *getPositivePair(conv.getStrides());
    auto dilations = *getPositivePair(conv.getDilations());
    for (int64_t value : {strides[0], strides[1], dilations[0], dilations[1]}) {
      operands.push_back(
          arith::ConstantIndexOp::create(rewriter, conv.getLoc(), value));
    }
    func::CallOp::create(rewriter, conv.getLoc(), runtimeFunction, operands);
    rewriter.eraseOp(conv);
  }
}

} // namespace mlir::tutorial
