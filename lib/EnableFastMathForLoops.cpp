#include "lib/EnableFastMathForLoops.h"

using namespace mlir;

namespace mlir::tutorial {

void EnableFastMathForLoopsPass::runOnOperation() {
  // `reassoc` is the flag LLVM's loop vectorizer checks before it will
  // reorder a floating-point reduction; `contract` additionally permits
  // fusing a multiply+add pair into a single FMA once vectorized.
  constexpr arith::FastMathFlags flags =
      arith::FastMathFlags::reassoc | arith::FastMathFlags::contract;
  auto attr = arith::FastMathFlagsAttr::get(&getContext(), flags);

  getOperation().walk([&](Operation *op) {
    if (auto mulf = dyn_cast<arith::MulFOp>(op))
      mulf.setFastmathAttr(attr);
    else if (auto addf = dyn_cast<arith::AddFOp>(op))
      addf.setFastmathAttr(attr);
  });
}

} // namespace mlir::tutorial
