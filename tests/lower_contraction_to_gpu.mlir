// RUN: tutorial-opt %s -lower-contraction-to-gpu="strategy=shared-fp32 target=sm_89 block-m=64 block-n=64 block-k=16 threads=256" | FileCheck %s

func.func @shared_matmul(%lhs: memref<65x17xf32>,
                         %rhs: memref<17x67xf32>,
                         %out: memref<65x67xf32>) {
  linalg.matmul ins(%lhs, %rhs : memref<65x17xf32>, memref<17x67xf32>)
                outs(%out : memref<65x67xf32>)
  return
}

// CHECK-LABEL: func.func @shared_matmul
// CHECK: gpu.launch blocks{{.*}} threads{{.*}} workgroup(%[[A:.*]] : memref<64x16xf32, #gpu.address_space<workgroup>>, %[[B:.*]] : memref<16x64xf32, #gpu.address_space<workgroup>>)
// CHECK: scf.for
// CHECK: memref.store {{.*}}, %[[A]]
// CHECK: scf.for
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
