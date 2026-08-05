// RUN: tutorial-opt %s -lower-contraction-to-gpu="strategy=shared-fp32 target=sm_89 block-m=64 block-n=64 block-k=16 threads=256" | FileCheck %s
// RUN: tutorial-opt %s -lower-contraction-to-gpu="strategy=shared-fp32 target=sm_80 block-m=64 block-n=64 block-k=16 threads=256" | FileCheck %s --check-prefix=TARGET-FALLBACK

func.func @shared_matmul(%lhs: memref<65x17xf32>,
                         %rhs: memref<17x67xf32>,
                         %out: memref<65x67xf32>) {
  linalg.matmul ins(%lhs, %rhs : memref<65x17xf32>, memref<17x67xf32>)
                outs(%out : memref<65x67xf32>)
  return
}

// CHECK-LABEL: func.func @shared_matmul
// CHECK: gpu.launch blocks{{.*}} threads{{.*}} workgroup(%[[A:.*]] : memref<64x16xf32, #gpu.address_space<workgroup>>, %[[B:.*]] : memref<16x65xf32, #gpu.address_space<workgroup>>)
// CHECK: vector.load {{.*}} : memref<65x17xf32>, vector<4xf32>
// CHECK: vector.store {{.*}}, %[[A]]{{.*}} : memref<64x16xf32, #gpu.address_space<workgroup>>, vector<4xf32>
// CHECK: memref.store {{.*}}, %[[A]]
// CHECK: vector.load {{.*}} : memref<17x67xf32>, vector<4xf32>
// CHECK: vector.store {{.*}}, %[[B]]{{.*}} : memref<16x65xf32, #gpu.address_space<workgroup>>, vector<4xf32>
// CHECK: memref.store {{.*}}, %[[B]]
// CHECK: gpu.barrier
// CHECK: scf.for
// CHECK: memref.load %[[A]]
// CHECK: memref.load %[[B]]
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: gpu.barrier
// CHECK: memref.store
// CHECK: gpu.terminator
// CHECK-NOT: linalg.matmul

func.func @unit_inner_stride_supported(
    %lhs: memref<65x17xf32, strided<[?, 1], offset: ?>>,
    %rhs: memref<17x67xf32, strided<[?, 1], offset: ?>>,
    %out: memref<65x67xf32, strided<[?, 1], offset: ?>>) {
  linalg.matmul
      ins(%lhs, %rhs : memref<65x17xf32, strided<[?, 1], offset: ?>>,
                       memref<17x67xf32, strided<[?, 1], offset: ?>>)
      outs(%out : memref<65x67xf32, strided<[?, 1], offset: ?>>)
  return
}

// CHECK-LABEL: func.func @unit_inner_stride_supported
// CHECK: gpu.launch
// CHECK-NOT: linalg.matmul

func.func @strided_fallback(
    %lhs: memref<65x17xf32, strided<[?, ?], offset: ?>>,
    %rhs: memref<17x67xf32, strided<[?, ?], offset: ?>>,
    %out: memref<65x67xf32, strided<[?, ?], offset: ?>>) {
  linalg.matmul
      ins(%lhs, %rhs : memref<65x17xf32, strided<[?, ?], offset: ?>>,
                       memref<17x67xf32, strided<[?, ?], offset: ?>>)
      outs(%out : memref<65x67xf32, strided<[?, ?], offset: ?>>)
  return
}

// CHECK-LABEL: func.func @strided_fallback
// CHECK: linalg.matmul

func.func @unsupported_f64(%lhs: memref<16x16xf64>,
                           %rhs: memref<16x16xf64>,
                           %out: memref<16x16xf64>) {
  linalg.matmul ins(%lhs, %rhs : memref<16x16xf64>, memref<16x16xf64>)
                outs(%out : memref<16x16xf64>)
  return
}

// CHECK-LABEL: func.func @unsupported_f64
// CHECK: linalg.matmul

// TARGET-FALLBACK-LABEL: func.func @shared_matmul
// TARGET-FALLBACK: linalg.matmul
// TARGET-FALLBACK-LABEL: func.func @unit_inner_stride_supported
// TARGET-FALLBACK: linalg.matmul
// TARGET-FALLBACK-LABEL: func.func @strided_fallback
// TARGET-FALLBACK: linalg.matmul
// TARGET-FALLBACK-LABEL: func.func @unsupported_f64
// TARGET-FALLBACK: linalg.matmul
// TARGET-FALLBACK-NOT: gpu.launch
