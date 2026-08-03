#include "lib/TileBatchMatMulForGpu.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;

namespace mlir::tutorial {
namespace {

bool isSupportedF32Tensor(Value value) {
  auto type = dyn_cast<MemRefType>(value.getType());
  return type && type.getRank() == 3 && type.getElementType().isF32();
}

bool isSupportedBatchMatmul(linalg::BatchMatmulOp batchMatmul) {
  if (batchMatmul->getNumResults() != 0 ||
      !linalg::BatchMatmulOp::isDefaultIndexingMaps(
          batchMatmul.getIndexingMapsAttr()))
    return false;

  auto inputs = batchMatmul.getDpsInputs();
  auto outputs = batchMatmul.getDpsInits();
  return inputs.size() == 2 && outputs.size() == 1 &&
         isSupportedF32Tensor(inputs[0]) && isSupportedF32Tensor(inputs[1]) &&
         isSupportedF32Tensor(outputs[0]);
}

} // namespace

void TileBatchMatMulForGpuPass::runOnOperation() {
  int64_t blockMInt = blockM;
  int64_t blockNInt = blockN;
  if (blockMInt <= 0 || blockNInt <= 0 || blockMInt > 1024 / blockNInt) {
    getOperation().emitError()
        << "GPU block dimensions must be positive and contain at most 1024 "
           "threads, got block-m="
        << blockMInt << " block-n=" << blockNInt;
    signalPassFailure();
    return;
  }

  SmallVector<linalg::BatchMatmulOp> worklist;
  getOperation().walk([&](linalg::BatchMatmulOp batchMatmul) {
    if (isSupportedBatchMatmul(batchMatmul))
      worklist.push_back(batchMatmul);
  });

  IRRewriter rewriter(&getContext());
  for (linalg::BatchMatmulOp batchMatmul : worklist) {
    Location loc = batchMatmul.getLoc();
    Value lhs = batchMatmul.getDpsInputs()[0];
    Value rhs = batchMatmul.getDpsInputs()[1];
    Value output = batchMatmul.getDpsInits()[0];

    rewriter.setInsertionPoint(batchMatmul);
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value blockMValue =
        arith::ConstantIndexOp::create(rewriter, loc, blockMInt);
    Value blockNValue =
        arith::ConstantIndexOp::create(rewriter, loc, blockNInt);
    Value batchSize = memref::DimOp::create(rewriter, loc, output, 0);
    Value mSize = memref::DimOp::create(rewriter, loc, output, 1);
    Value nSize = memref::DimOp::create(rewriter, loc, output, 2);
    Value kSize = memref::DimOp::create(rewriter, loc, lhs, 2);

    scf::ParallelOp::create(
        rewriter, loc, ValueRange{zero, zero, zero},
        ValueRange{batchSize, mSize, nSize},
        ValueRange{one, blockMValue, blockNValue},
        [&](OpBuilder &blockBuilder, Location blockLoc,
            ValueRange blockIndices) {
          scf::ParallelOp::create(
              blockBuilder, blockLoc, ValueRange{zero, zero},
              ValueRange{blockMValue, blockNValue}, ValueRange{one, one},
              [&](OpBuilder &threadBuilder, Location threadLoc,
                  ValueRange threadIndices) {
                Value row =
                    arith::AddIOp::create(threadBuilder, threadLoc,
                                          blockIndices[1], threadIndices[0]);
                Value column =
                    arith::AddIOp::create(threadBuilder, threadLoc,
                                          blockIndices[2], threadIndices[1]);
                Value rowInBounds = arith::CmpIOp::create(
                    threadBuilder, threadLoc, arith::CmpIPredicate::ult, row,
                    mSize);
                Value columnInBounds = arith::CmpIOp::create(
                    threadBuilder, threadLoc, arith::CmpIPredicate::ult, column,
                    nSize);
                Value inBounds = arith::AndIOp::create(
                    threadBuilder, threadLoc, rowInBounds, columnInBounds);

                scf::IfOp::create(
                    threadBuilder, threadLoc, inBounds,
                    [&](OpBuilder &ifBuilder, Location ifLoc) {
                      Value initial = memref::LoadOp::create(
                          ifBuilder, ifLoc, output,
                          ValueRange{blockIndices[0], row, column});
                      auto reduction = scf::ForOp::create(
                          ifBuilder, ifLoc, zero, kSize, one,
                          ValueRange{initial},
                          [&](OpBuilder &reductionBuilder,
                              Location reductionLoc, Value k,
                              ValueRange iterArgs) {
                            Value lhsValue = memref::LoadOp::create(
                                reductionBuilder, reductionLoc, lhs,
                                ValueRange{blockIndices[0], row, k});
                            Value rhsValue = memref::LoadOp::create(
                                reductionBuilder, reductionLoc, rhs,
                                ValueRange{blockIndices[0], k, column});
                            Value product = arith::MulFOp::create(
                                reductionBuilder, reductionLoc, lhsValue,
                                rhsValue);
                            Value sum = arith::AddFOp::create(
                                reductionBuilder, reductionLoc, iterArgs[0],
                                product);
                            scf::YieldOp::create(reductionBuilder, reductionLoc,
                                                 sum);
                          });
                      memref::StoreOp::create(
                          ifBuilder, ifLoc, reduction.getResult(0), output,
                          ValueRange{blockIndices[0], row, column});
                      scf::YieldOp::create(ifBuilder, ifLoc);
                    });
              });
        });

    rewriter.eraseOp(batchMatmul);
  }
}

} // namespace mlir::tutorial
