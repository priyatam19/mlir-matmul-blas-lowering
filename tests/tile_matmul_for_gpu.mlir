// RUN: tutorial-opt %s -tile-matmul-for-gpu="block-m=8 block-n=32" | FileCheck %s

func.func @static_matmul(%lhs: memref<33x17xf32>,
                         %rhs: memref<17x65xf32>,
                         %out: memref<33x65xf32>) {
  linalg.matmul ins(%lhs, %rhs : memref<33x17xf32>, memref<17x65xf32>)
                outs(%out : memref<33x65xf32>)
  return
}

// CHECK-LABEL: func.func @static_matmul
// CHECK: %[[M:.*]] = memref.dim %{{.*}}, %c0
// CHECK: %[[N:.*]] = memref.dim %{{.*}}, %c1
// CHECK: scf.parallel (%[[BM:.*]], %[[BN:.*]]) = {{.*}} to (%[[M]], %[[N]]) step (%c8, %c32)
// CHECK: scf.parallel (%[[TM:.*]], %[[TN:.*]]) = {{.*}} to (%c8, %c32) step (%c1, %c1)
// CHECK: %[[ROW:.*]] = arith.addi %[[BM]], %[[TM]]
// CHECK: %[[COL:.*]] = arith.addi %[[BN]], %[[TN]]
// CHECK: arith.cmpi ult, %[[ROW]], %[[M]]
// CHECK: arith.cmpi ult, %[[COL]], %[[N]]
// CHECK: scf.if
// CHECK: %[[INITIAL:.*]] = memref.load %{{.*}}[%[[ROW]], %[[COL]]]
// CHECK: %[[SUM:.*]] = scf.for
// CHECK-SAME: iter_args(%{{.*}} = %[[INITIAL]]) -> (f32)
// CHECK: memref.load %{{.*}}[%[[ROW]], %{{.*}}]
// CHECK: memref.load %{{.*}}[%{{.*}}, %[[COL]]]
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: scf.yield
// CHECK: memref.store %[[SUM]], %{{.*}}[%[[ROW]], %[[COL]]]
// CHECK-NOT: linalg.matmul

func.func @dynamic_matmul(%lhs: memref<?x?xf32>,
                          %rhs: memref<?x?xf32>,
                          %out: memref<?x?xf32>) {
  linalg.matmul ins(%lhs, %rhs : memref<?x?xf32>, memref<?x?xf32>)
                outs(%out : memref<?x?xf32>)
  return
}

// CHECK-LABEL: func.func @dynamic_matmul
// CHECK: scf.parallel
// CHECK: scf.parallel
// CHECK: scf.if
// CHECK: scf.for
// CHECK-NOT: linalg.matmul

func.func @strided_matmul(
    %lhs: memref<?x?xf32, strided<[?, ?], offset: ?>>,
    %rhs: memref<?x?xf32, strided<[?, ?], offset: ?>>,
    %out: memref<?x?xf32, strided<[?, ?], offset: ?>>) {
  linalg.matmul
      ins(%lhs, %rhs : memref<?x?xf32, strided<[?, ?], offset: ?>>,
                        memref<?x?xf32, strided<[?, ?], offset: ?>>)
      outs(%out : memref<?x?xf32, strided<[?, ?], offset: ?>>)
  return
}

// CHECK-LABEL: func.func @strided_matmul
// CHECK: scf.parallel
// CHECK: scf.parallel
// CHECK: scf.if
// CHECK: scf.for
// CHECK-NOT: linalg.matmul

func.func @mixed_linalg(%lhs: memref<8x8xf32>, %rhs: memref<8x8xf32>,
                        %out: memref<8x8xf32>, %zero: f32) {
  linalg.fill ins(%zero : f32) outs(%out : memref<8x8xf32>)
  linalg.matmul ins(%lhs, %rhs : memref<8x8xf32>, memref<8x8xf32>)
                outs(%out : memref<8x8xf32>)
  return
}

// CHECK-LABEL: func.func @mixed_linalg
// CHECK: linalg.fill
// CHECK: scf.parallel
// CHECK-NOT: linalg.matmul

func.func @unsupported_f64(%lhs: memref<8x8xf64>, %rhs: memref<8x8xf64>,
                           %out: memref<8x8xf64>) {
  linalg.matmul ins(%lhs, %rhs : memref<8x8xf64>, memref<8x8xf64>)
                outs(%out : memref<8x8xf64>)
  return
}

// CHECK-LABEL: func.func @unsupported_f64
// CHECK: linalg.matmul
