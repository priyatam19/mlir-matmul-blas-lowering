// RUN: tutorial-opt %s -lower-contraction-to-gpu="strategy=tensorcore-tf32 target=sm_89 block-m=64 block-n=64 block-k=16 threads=256 vector-width=4 stages=1" | FileCheck %s

func.func @tf32_matmul(%lhs: memref<65x17xf32>,
                       %rhs: memref<17x67xf32>,
                       %out: memref<65x67xf32>) {
  linalg.matmul ins(%lhs, %rhs : memref<65x17xf32>, memref<17x67xf32>)
                outs(%out : memref<65x67xf32>)
  return
}

// CHECK-LABEL: func.func @tf32_matmul
// CHECK: gpu.launch {{.*}} workgroup(%[[A:.*]] : memref<64x16xf32, #gpu.address_space<workgroup>>, %[[B:.*]] : memref<16x64xf32, #gpu.address_space<workgroup>>)
// CHECK: vector.load
// CHECK: vector.store
// CHECK: gpu.barrier
// CHECK: nvgpu.ldmatrix %[[A]]
// CHECK: memref.load %[[B]]
// CHECK: nvgpu.mma.sync
// CHECK-SAME: mmaShape = [16, 8, 8]
// CHECK-SAME: tf32Enabled
// CHECK: gpu.barrier
// CHECK: vector.extract
// CHECK: memref.store
// CHECK-NOT: linalg.matmul
