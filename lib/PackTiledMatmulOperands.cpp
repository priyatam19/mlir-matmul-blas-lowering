#include "lib/PackTiledMatmulOperands.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;

namespace mlir::tutorial {
namespace {

// memref.copy lowers to a runtime call (memrefCopy) rather than an inline
// loop -- both an extra link dependency this project's CPU path doesn't
// otherwise have, and, at the tile-call counts --tile-matmul-for-cache
// produces (millions, at cache-exceeding scale), an opaque per-call
// function-call cost for what should be a few dozen inlined, vectorizable
// loads/stores. Hand-rolling the copy as a loop avoids both; matmul
// operands are always rank 2, so a fixed nested loop is enough.
void emitCopyLoop(OpBuilder &builder, Location loc, Value src, Value dst) {
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value rows = builder.create<memref::DimOp>(loc, src, 0);
  Value cols = builder.create<memref::DimOp>(loc, src, 1);
  builder.create<scf::ForOp>(
      loc, zero, rows, one, ValueRange{},
      [&](OpBuilder &rowBuilder, Location rowLoc, Value row, ValueRange) {
        rowBuilder.create<scf::ForOp>(
            rowLoc, zero, cols, one, ValueRange{},
            [&](OpBuilder &colBuilder, Location colLoc, Value col,
                ValueRange) {
              Value v = colBuilder.create<memref::LoadOp>(
                  colLoc, src, ValueRange{row, col});
              colBuilder.create<memref::StoreOp>(colLoc, v, dst,
                                                 ValueRange{row, col});
              colBuilder.create<scf::YieldOp>(colLoc);
            });
        rowBuilder.create<scf::YieldOp>(rowLoc);
      });
}

} // namespace

void PackTiledMatmulOperandsPass::runOnOperation() {
  ModuleOp module = getOperation();
  IRRewriter rewriter(module.getContext());

  SmallVector<linalg::MatmulOp> worklist;
  module.walk([&](linalg::MatmulOp matmul) {
    // Bufferized (memref-semantic) matmuls only -- this pass runs after
    // --linalg-to-bufferization.
    if (matmul->getNumResults() == 0)
      worklist.push_back(matmul);
  });

  for (linalg::MatmulOp matmul : worklist) {
    Location loc = matmul.getLoc();

    // A stack allocation inside a loop body only gets reused across
    // iterations (instead of growing the stack once per iteration -- fatal
    // at the millions of tile-calls cache-exceeding scales produce) if the
    // backend hoists it out on its own, which it does not reliably do here.
    // Hoist it explicitly to just before the outermost enclosing scf.for so
    // one buffer is allocated once and refilled every iteration, matching
    // how a real microkernel's scratch space works. Sizes here are static
    // (fixed tile dimensions) in every case this project currently
    // benchmarks; a dynamic size (a ragged tail tile) can't be computed
    // before the loop that produces it exists, so that rarer case falls
    // back to allocating at the matmul's own site instead.
    Operation *hoistPoint = matmul;
    for (Operation *parent = matmul->getParentOp();
        isa_and_nonnull<scf::ForOp>(parent); parent = parent->getParentOp())
      hoistPoint = parent;

    for (OpOperand *input : matmul.getDpsInputOperands()) {
      Value operand = input->get();
      if (!operand.getDefiningOp<memref::SubViewOp>())
        continue; // Not a strided tile view -- nothing to pack.

      auto memrefType = cast<MemRefType>(operand.getType());
      auto packedType =
          MemRefType::get(memrefType.getShape(), memrefType.getElementType());

      Value packed;
      if (packedType.hasStaticShape()) {
        rewriter.setInsertionPoint(hoistPoint);
        packed = rewriter.create<memref::AllocaOp>(loc, packedType);
      } else {
        rewriter.setInsertionPoint(matmul);
        SmallVector<Value> dynamicSizes;
        for (int64_t dim = 0; dim < memrefType.getRank(); ++dim)
          if (memrefType.isDynamicDim(dim))
            dynamicSizes.push_back(
                rewriter.create<memref::DimOp>(loc, operand, dim));
        packed =
            rewriter.create<memref::AllocaOp>(loc, packedType, dynamicSizes);
      }

      rewriter.setInsertionPoint(matmul);
      emitCopyLoop(rewriter, loc, operand, packed);
      matmul->setOperand(input->getOperandNumber(), packed);
    }
  }
}

} // namespace mlir::tutorial
