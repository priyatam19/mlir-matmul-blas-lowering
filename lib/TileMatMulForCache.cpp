#include "lib/TileMatMulForCache.h"

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;

namespace mlir::tutorial {

void TileMatMulForCachePass::runOnOperation() {
  ModuleOp module = getOperation();
  IRRewriter rewriter(module.getContext());
  SmallVector<linalg::MatmulOp> worklist;

  module.walk([&](linalg::MatmulOp matmul) {
    // Keep bufferized matmuls available for the OpenBLAS lowering pass. This
    // pass is meant to demonstrate tensor-level tiling before bufferization.
    if (matmul->getNumResults() == 0)
      return;
    worklist.push_back(matmul);
  });

  bool changed = false;
  for (linalg::MatmulOp matmul : worklist) {
    if (matmul->use_empty())
      continue;

    rewriter.setInsertionPoint(matmul);
    linalg::LinalgTilingOptions options;
    options.setTileSizes({tileM, tileN, tileK});
    options.setLoopType(linalg::LinalgTilingLoopType::Loops);

    FailureOr<linalg::TiledLinalgOp> tiled = linalg::tileLinalgOp(
        rewriter, cast<linalg::LinalgOp>(matmul.getOperation()), options);
    if (failed(tiled)) {
      matmul.emitError("failed to tile linalg.matmul");
      signalPassFailure();
      return;
    }

    rewriter.replaceOp(matmul, tiled->tensorResults);
    changed = true;
  }

  if (!changed)
    return;
}

} // namespace mlir::tutorial
