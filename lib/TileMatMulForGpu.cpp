#include "lib/TileMatMulForGpu.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;

namespace mlir::tutorial {
namespace {

bool isSupportedF32Matrix(Value value) {
  auto type = dyn_cast<MemRefType>(value.getType());
  return type && type.getRank() == 2 && type.getElementType().isF32();
}

} // namespace

void TileMatMulForGpuPass::runOnOperation() {
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

  SmallVector<linalg::MatmulOp> worklist;
  getOperation().walk([&](linalg::MatmulOp matmul) {
    if (matmul->getNumResults() != 0)
      return;

    auto inputs = matmul.getDpsInputs();
    auto outputs = matmul.getDpsInits();
    if (inputs.size() != 2 || outputs.size() != 1)
      return;
    if (!isSupportedF32Matrix(inputs[0]) || !isSupportedF32Matrix(inputs[1]) ||
        !isSupportedF32Matrix(outputs[0]))
      return;
    worklist.push_back(matmul);
  });

  IRRewriter rewriter(&getContext());
  for (linalg::MatmulOp matmul : worklist) {
    Location loc = matmul.getLoc();
    Value lhs = matmul.getDpsInputs()[0];
    Value rhs = matmul.getDpsInputs()[1];
    Value output = matmul.getDpsInits()[0];

    rewriter.setInsertionPoint(matmul);
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value blockMValue =
        arith::ConstantIndexOp::create(rewriter, loc, blockMInt);
    Value blockNValue =
        arith::ConstantIndexOp::create(rewriter, loc, blockNInt);
    Value mSize = memref::DimOp::create(rewriter, loc, output, 0);
    Value nSize = memref::DimOp::create(rewriter, loc, output, 1);
    Value kSize = memref::DimOp::create(rewriter, loc, lhs, 1);

    scf::ParallelOp::create(
        rewriter, loc, ValueRange{zero, zero}, ValueRange{mSize, nSize},
        ValueRange{blockMValue, blockNValue},
        [&](OpBuilder &blockBuilder, Location blockLoc,
            ValueRange blockIndices) {
          scf::ParallelOp::create(
              blockBuilder, blockLoc, ValueRange{zero, zero},
              ValueRange{blockMValue, blockNValue}, ValueRange{one, one},
              [&](OpBuilder &threadBuilder, Location threadLoc,
                  ValueRange threadIndices) {
                Value row =
                    arith::AddIOp::create(threadBuilder, threadLoc,
                                          blockIndices[0], threadIndices[0]);
                Value column =
                    arith::AddIOp::create(threadBuilder, threadLoc,
                                          blockIndices[1], threadIndices[1]);
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
                          ifBuilder, ifLoc, output, ValueRange{row, column});
                      auto reduction = scf::ForOp::create(
                          ifBuilder, ifLoc, zero, kSize, one,
                          ValueRange{initial},
                          [&](OpBuilder &reductionBuilder,
                              Location reductionLoc, Value k,
                              ValueRange iterArgs) {
                            Value lhsValue = memref::LoadOp::create(
                                reductionBuilder, reductionLoc, lhs,
                                ValueRange{row, k});
                            Value rhsValue = memref::LoadOp::create(
                                reductionBuilder, reductionLoc, rhs,
                                ValueRange{k, column});
                            Value product = arith::MulFOp::create(
                                reductionBuilder, reductionLoc, lhsValue,
                                rhsValue);
                            Value sum = arith::AddFOp::create(
                                reductionBuilder, reductionLoc, iterArgs[0],
                                product);
                            scf::YieldOp::create(reductionBuilder, reductionLoc,
                                                 sum);
                          });
                      memref::StoreOp::create(ifBuilder, ifLoc,
                                              reduction.getResult(0), output,
                                              ValueRange{row, column});
                      scf::YieldOp::create(ifBuilder, ifLoc);
                    });
              });
        });

    rewriter.eraseOp(matmul);
  }
}

} // namespace mlir::tutorial
