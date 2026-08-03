#include "lib/TileConv2DNchwForGpu.h"

#include "mlir/IR/PatternMatch.h"

#include <array>

using namespace mlir;

namespace mlir::tutorial {
namespace {

bool isF32Rank4MemRef(Value value) {
  auto type = dyn_cast<MemRefType>(value.getType());
  return type && type.getRank() == 4 && type.getElementType().isF32();
}

FailureOr<std::array<int64_t, 2>> getPositivePair(DenseIntElementsAttr attr) {
  if (!attr || attr.getNumElements() != 2)
    return failure();
  std::array<int64_t, 2> values;
  size_t index = 0;
  for (APInt value : attr.getValues<APInt>()) {
    values[index++] = value.getSExtValue();
  }
  if (values[0] <= 0 || values[1] <= 0)
    return failure();
  return values;
}

bool isSupportedConv(linalg::Conv2DNchwFchwOp conv) {
  if (conv->getNumResults() != 0 ||
      failed(getPositivePair(conv.getStrides())) ||
      failed(getPositivePair(conv.getDilations())))
    return false;
  auto inputs = conv.getDpsInputs();
  auto outputs = conv.getDpsInits();
  return inputs.size() == 2 && outputs.size() == 1 &&
         isF32Rank4MemRef(inputs[0]) && isF32Rank4MemRef(inputs[1]) &&
         isF32Rank4MemRef(outputs[0]);
}

Value multiply(OpBuilder &builder, Location loc, Value lhs, Value rhs) {
  return arith::MulIOp::create(builder, loc, lhs, rhs);
}

} // namespace

void TileConv2DNchwForGpuPass::runOnOperation() {
  int64_t threadsInt = threads;
  if (threadsInt <= 0 || threadsInt > 1024) {
    getOperation().emitError()
        << "GPU convolution thread count must be between 1 and 1024, got "
        << threadsInt;
    signalPassFailure();
    return;
  }

  SmallVector<linalg::Conv2DNchwFchwOp> worklist;
  getOperation().walk([&](linalg::Conv2DNchwFchwOp conv) {
    if (isSupportedConv(conv))
      worklist.push_back(conv);
  });

  IRRewriter rewriter(&getContext());
  for (linalg::Conv2DNchwFchwOp conv : worklist) {
    Location loc = conv.getLoc();
    Value input = conv.getDpsInputs()[0];
    Value filter = conv.getDpsInputs()[1];
    Value output = conv.getDpsInits()[0];
    auto strides = *getPositivePair(conv.getStrides());
    auto dilations = *getPositivePair(conv.getDilations());

    rewriter.setInsertionPoint(conv);
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value threadsValue =
        arith::ConstantIndexOp::create(rewriter, loc, threadsInt);
    Value strideH = arith::ConstantIndexOp::create(rewriter, loc, strides[0]);
    Value strideW = arith::ConstantIndexOp::create(rewriter, loc, strides[1]);
    Value dilationH =
        arith::ConstantIndexOp::create(rewriter, loc, dilations[0]);
    Value dilationW =
        arith::ConstantIndexOp::create(rewriter, loc, dilations[1]);

    Value nSize = memref::DimOp::create(rewriter, loc, output, 0);
    Value fSize = memref::DimOp::create(rewriter, loc, output, 1);
    Value ohSize = memref::DimOp::create(rewriter, loc, output, 2);
    Value owSize = memref::DimOp::create(rewriter, loc, output, 3);
    Value cSize = memref::DimOp::create(rewriter, loc, input, 1);
    Value khSize = memref::DimOp::create(rewriter, loc, filter, 2);
    Value kwSize = memref::DimOp::create(rewriter, loc, filter, 3);
    Value total = multiply(rewriter, loc, nSize, fSize);
    total = multiply(rewriter, loc, total, ohSize);
    total = multiply(rewriter, loc, total, owSize);

    scf::ParallelOp::create(
        rewriter, loc, ValueRange{zero}, ValueRange{total},
        ValueRange{threadsValue},
        [&](OpBuilder &blockBuilder, Location blockLoc,
            ValueRange blockIndices) {
          scf::ParallelOp::create(
              blockBuilder, blockLoc, ValueRange{zero},
              ValueRange{threadsValue}, ValueRange{one},
              [&](OpBuilder &threadBuilder, Location threadLoc,
                  ValueRange threadIndices) {
                Value linear =
                    arith::AddIOp::create(threadBuilder, threadLoc,
                                          blockIndices[0], threadIndices[0]);
                Value inBounds = arith::CmpIOp::create(
                    threadBuilder, threadLoc, arith::CmpIPredicate::ult, linear,
                    total);
                scf::IfOp::create(
                    threadBuilder, threadLoc, inBounds,
                    [&](OpBuilder &ifBuilder, Location ifLoc) {
                      Value ow = arith::RemUIOp::create(ifBuilder, ifLoc,
                                                        linear, owSize);
                      Value quotient = arith::DivUIOp::create(ifBuilder, ifLoc,
                                                              linear, owSize);
                      Value oh = arith::RemUIOp::create(ifBuilder, ifLoc,
                                                        quotient, ohSize);
                      quotient = arith::DivUIOp::create(ifBuilder, ifLoc,
                                                        quotient, ohSize);
                      Value f = arith::RemUIOp::create(ifBuilder, ifLoc,
                                                       quotient, fSize);
                      Value n = arith::DivUIOp::create(ifBuilder, ifLoc,
                                                       quotient, fSize);
                      Value initial = memref::LoadOp::create(
                          ifBuilder, ifLoc, output, ValueRange{n, f, oh, ow});

                      auto cLoop = scf::ForOp::create(
                          ifBuilder, ifLoc, zero, cSize, one,
                          ValueRange{initial},
                          [&](OpBuilder &cBuilder, Location cLoc, Value c,
                              ValueRange cArgs) {
                            auto khLoop = scf::ForOp::create(
                                cBuilder, cLoc, zero, khSize, one, cArgs,
                                [&](OpBuilder &khBuilder, Location khLoc,
                                    Value kh, ValueRange khArgs) {
                                  auto kwLoop = scf::ForOp::create(
                                      khBuilder, khLoc, zero, kwSize, one,
                                      khArgs,
                                      [&](OpBuilder &kwBuilder, Location kwLoc,
                                          Value kw, ValueRange kwArgs) {
                                        Value inputH = multiply(
                                            kwBuilder, kwLoc, oh, strideH);
                                        inputH = arith::AddIOp::create(
                                            kwBuilder, kwLoc, inputH,
                                            multiply(kwBuilder, kwLoc, kh,
                                                     dilationH));
                                        Value inputW = multiply(
                                            kwBuilder, kwLoc, ow, strideW);
                                        inputW = arith::AddIOp::create(
                                            kwBuilder, kwLoc, inputW,
                                            multiply(kwBuilder, kwLoc, kw,
                                                     dilationW));
                                        Value inputValue =
                                            memref::LoadOp::create(
                                                kwBuilder, kwLoc, input,
                                                ValueRange{n, c, inputH,
                                                           inputW});
                                        Value filterValue =
                                            memref::LoadOp::create(
                                                kwBuilder, kwLoc, filter,
                                                ValueRange{f, c, kh, kw});
                                        Value product = arith::MulFOp::create(
                                            kwBuilder, kwLoc, inputValue,
                                            filterValue);
                                        Value sum = arith::AddFOp::create(
                                            kwBuilder, kwLoc, kwArgs[0],
                                            product);
                                        scf::YieldOp::create(kwBuilder, kwLoc,
                                                             sum);
                                      });
                                  scf::YieldOp::create(khBuilder, khLoc,
                                                       kwLoop.getResult(0));
                                });
                            scf::YieldOp::create(cBuilder, cLoc,
                                                 khLoop.getResult(0));
                          });
                      memref::StoreOp::create(ifBuilder, ifLoc,
                                              cLoop.getResult(0), output,
                                              ValueRange{n, f, oh, ow});
                      scf::YieldOp::create(ifBuilder, ifLoc);
                    });
              });
        });
    rewriter.eraseOp(conv);
  }
}

} // namespace mlir::tutorial
