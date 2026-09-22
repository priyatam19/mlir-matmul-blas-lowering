// RUN: tutorial-opt %s -tile-matmul-for-cache="tile-m=32 tile-n=32 tile-k=32" -linalg-to-bufferization -pack-tiled-matmul-operands | FileCheck %s

// A tile view and a packed copy of it can both fit a cache level in raw
// byte count, but the view's rows are strided by the *original* matrix's
// width -- see lib/PackTiledMatmulOperands.h for why that still costs real
// performance (cache-set-conflict aliasing, one TLB entry per row instead
// of one for the whole tile), and docs/cpu_matmul_blas_ablation.md Result 6
// for the measured effect (1.65x on top of tiling alone).

func.func @mm(%a: tensor<64x64xf32>, %b: tensor<64x64xf32>,
             %c: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %r = linalg.matmul ins(%a, %b : tensor<64x64xf32>, tensor<64x64xf32>)
                     outs(%c : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %r : tensor<64x64xf32>
}

// The two alloca buffers are hoisted before the tile loop nest and reused
// every iteration -- allocating fresh inside the loop would grow the stack
// once per tile-call, millions of times at cache-exceeding scale, and
// segfault (see docs/cpu_matmul_blas_ablation.md Result 6 for how this was
// found).
// CHECK-LABEL: func.func @mm
// CHECK: %[[A_PACKED:.*]] = memref.alloca() : memref<32x32xf32>
// CHECK: %[[B_PACKED:.*]] = memref.alloca() : memref<32x32xf32>
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK:       %[[A_TILE:.*]] = memref.subview
// CHECK:       %[[B_TILE:.*]] = memref.subview
// CHECK:       %[[C_TILE:.*]] = memref.subview
// CHECK:       scf.for
// CHECK:         scf.for
// CHECK:           %[[AV:.*]] = memref.load %[[A_TILE]]
// CHECK:           memref.store %[[AV]], %[[A_PACKED]]
// CHECK:       scf.for
// CHECK:         scf.for
// CHECK:           %[[BV:.*]] = memref.load %[[B_TILE]]
// CHECK:           memref.store %[[BV]], %[[B_PACKED]]
// CHECK:       linalg.matmul ins(%[[A_PACKED]], %[[B_PACKED]] : memref<32x32xf32>, memref<32x32xf32>) outs(%[[C_TILE]]
