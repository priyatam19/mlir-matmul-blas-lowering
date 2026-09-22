#ifndef MLIR_TUTORIAL_PACK_TILED_MATMUL_OPERANDS_H
#define MLIR_TUTORIAL_PACK_TILED_MATMUL_OPERANDS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"

namespace mlir::tutorial {

// Runs post-bufferization, after --tile-matmul-for-cache has produced a
// tiled loop nest of small linalg.matmul ops reading memref.subview tiles
// out of a much larger matrix. A tile view and a packed copy of that view
// can both fit a cache level in raw byte count, but the view's rows are
// strided by the *original* matrix's width -- large strides can alias in a
// cache's set associativity even when total capacity is respected, and each
// row lands on its own TLB entry instead of the whole tile sharing one.
// This pass copies each tiled linalg.matmul's input operands into a
// genuinely contiguous stack buffer (memref.alloca, sized once and reused
// every loop iteration -- these tiles are deliberately small, see
// --tile-matmul-for-cache's own doc comment) before the matmul runs.
//
// An earlier attempt did this at the tensor level, inside
// --tile-matmul-for-cache itself, via bufferization.alloc_tensor +
// tensor.insert_slice. One-Shot Bufferize recognized that pattern as a
// same-size, zero-offset, unit-stride copy -- semantically a no-op -- and
// optimized it away entirely, silently leaving the original strided
// subview in place. Operating on memrefs after bufferization has already
// run avoids that: there is no tensor-level dataflow analysis left to see
// through it.
struct PackTiledMatmulOperandsPass
    : public PassWrapper<PackTiledMatmulOperandsPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PackTiledMatmulOperandsPass)

  StringRef getArgument() const final {
    return "pack-tiled-matmul-operands";
  }

  StringRef getDescription() const final {
    return "Copy each linalg.matmul's memref.subview input operands into a "
           "contiguous stack buffer before the matmul runs";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() final;
};

} // namespace mlir::tutorial

#endif // MLIR_TUTORIAL_PACK_TILED_MATMUL_OPERANDS_H
