// RUN: tutorial-opt %s -lower-contraction-to-gpu="strategy=shared-fp32 target=sm_89 block-m=64 block-n=64 block-k=16 threads=256 vector-width=4 stages=2" | FileCheck %s

func.func @async_shared_matmul(%lhs: memref<65x17xf32>,
                               %rhs: memref<17x67xf32>,
                               %out: memref<65x67xf32>) {
  linalg.matmul ins(%lhs, %rhs : memref<65x17xf32>, memref<17x67xf32>)
                outs(%out : memref<65x67xf32>)
  return
}

// CHECK-LABEL: func.func @async_shared_matmul
// CHECK: gpu.launch {{.*}} workgroup(%[[A:.*]] : memref<2x64x16xf32, #gpu.address_space<workgroup>>, %[[B:.*]] : memref<2x16x64xf32, #gpu.address_space<workgroup>>)
// CHECK: nvgpu.device_async_copy {{.*}}, %[[A]]
// CHECK: nvgpu.device_async_copy {{.*}}, %[[B]]
// CHECK: nvgpu.device_async_create_group
// CHECK: nvgpu.device_async_wait
// CHECK: gpu.barrier
// CHECK: scf.for
// CHECK: nvgpu.device_async_copy
// CHECK: memref.load %[[A]]
// CHECK: memref.load %[[B]]
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: nvgpu.device_async_wait
// CHECK: gpu.barrier
// CHECK: memref.store
// CHECK-NOT: linalg.matmul
