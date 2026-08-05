// RUN: tutorial-opt %s -tile-batch-matmul-for-gpu="block-m=8 block-n=32" | FileCheck %s

#lhs_transpose = affine_map<(b, m, n, k) -> (b, k, m)>
#rhs_default = affine_map<(b, m, n, k) -> (b, k, n)>
#out_default = affine_map<(b, m, n, k) -> (b, m, n)>

func.func @static_batch_matmul(%lhs: memref<2x33x17xf32>,
                               %rhs: memref<2x17x65xf32>,
                               %out: memref<2x33x65xf32>) {
  linalg.batch_matmul
      ins(%lhs, %rhs : memref<2x33x17xf32>, memref<2x17x65xf32>)
      outs(%out : memref<2x33x65xf32>)
  return
}

// CHECK-LABEL: func.func @static_batch_matmul
// CHECK: %[[B:.*]] = memref.dim %{{.*}}, %c0
// CHECK: %[[M:.*]] = memref.dim %{{.*}}, %c1
// CHECK: %[[N:.*]] = memref.dim %{{.*}}, %c2
// CHECK: scf.parallel (%[[BB:.*]], %[[BM:.*]], %[[BN:.*]]) = {{.*}} to (%[[B]], %[[M]], %[[N]]) step (%c1, %c8, %c32)
// CHECK: scf.parallel (%[[TM:.*]], %[[TN:.*]]) = {{.*}} to (%c8, %c32) step (%c1, %c1)
// CHECK: %[[ROW:.*]] = arith.addi %[[BM]], %[[TM]]
// CHECK: %[[COL:.*]] = arith.addi %[[BN]], %[[TN]]
// CHECK: arith.cmpi ult, %[[ROW]], %[[M]]
// CHECK: arith.cmpi ult, %[[COL]], %[[N]]
// CHECK: scf.if
// CHECK: %[[INITIAL:.*]] = memref.load %{{.*}}[%[[BB]], %[[ROW]], %[[COL]]]
// CHECK: %[[SUM:.*]] = scf.for
// CHECK-SAME: iter_args(%{{.*}} = %[[INITIAL]]) -> (f32)
// CHECK: memref.load %{{.*}}[%[[BB]], %[[ROW]], %{{.*}}]
// CHECK: memref.load %{{.*}}[%[[BB]], %{{.*}}, %[[COL]]]
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: memref.store %[[SUM]], %{{.*}}[%[[BB]], %[[ROW]], %[[COL]]]
// CHECK-NOT: linalg.batch_matmul

func.func @dynamic_batch_matmul(%lhs: memref<?x?x?xf32>,
                                %rhs: memref<?x?x?xf32>,
                                %out: memref<?x?x?xf32>) {
  linalg.batch_matmul
      ins(%lhs, %rhs : memref<?x?x?xf32>, memref<?x?x?xf32>)
      outs(%out : memref<?x?x?xf32>)
  return
}

// CHECK-LABEL: func.func @dynamic_batch_matmul
// CHECK: scf.parallel
// CHECK: scf.parallel
// CHECK: scf.if
// CHECK: scf.for
// CHECK-NOT: linalg.batch_matmul

func.func @strided_batch_matmul(
    %lhs: memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>,
    %rhs: memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>,
    %out: memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>) {
  linalg.batch_matmul
      ins(%lhs, %rhs
          : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>,
            memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>)
      outs(%out : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>)
  return
}

// CHECK-LABEL: func.func @strided_batch_matmul
// CHECK: scf.parallel
// CHECK: scf.parallel
// CHECK-NOT: linalg.batch_matmul

func.func @mixed_linalg(%lhs: memref<2x8x8xf32>,
                        %rhs: memref<2x8x8xf32>,
                        %out: memref<2x8x8xf32>, %zero: f32) {
  linalg.fill ins(%zero : f32) outs(%out : memref<2x8x8xf32>)
  linalg.batch_matmul
      ins(%lhs, %rhs : memref<2x8x8xf32>, memref<2x8x8xf32>)
      outs(%out : memref<2x8x8xf32>)
  return
}

// CHECK-LABEL: func.func @mixed_linalg
// CHECK: linalg.fill
// CHECK: scf.parallel
// CHECK-NOT: linalg.batch_matmul

func.func @unsupported_f64(%lhs: memref<2x8x8xf64>,
                           %rhs: memref<2x8x8xf64>,
                           %out: memref<2x8x8xf64>) {
  linalg.batch_matmul
      ins(%lhs, %rhs : memref<2x8x8xf64>, memref<2x8x8xf64>)
      outs(%out : memref<2x8x8xf64>)
  return
}

// CHECK-LABEL: func.func @unsupported_f64
// CHECK: linalg.batch_matmul

func.func @unsupported_transpose(%lhs: memref<2x17x33xf32>,
                                 %rhs: memref<2x17x65xf32>,
                                 %out: memref<2x33x65xf32>) {
  linalg.batch_matmul
      indexing_maps = [#lhs_transpose, #rhs_default, #out_default]
      ins(%lhs, %rhs : memref<2x17x33xf32>, memref<2x17x65xf32>)
      outs(%out : memref<2x33x65xf32>)
  return
}

// CHECK-LABEL: func.func @unsupported_transpose
// CHECK: linalg.batch_matmul
