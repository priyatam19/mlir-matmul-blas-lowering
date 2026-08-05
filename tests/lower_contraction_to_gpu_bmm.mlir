// RUN: tutorial-opt %s -lower-contraction-to-gpu="strategy=shared-fp32 target=sm_89 block-m=64 block-n=64 block-k=16 threads=256 vector-width=4 stages=2" | FileCheck %s --check-prefix=SHARED
// RUN: tutorial-opt %s -lower-contraction-to-gpu="strategy=tensorcore-tf32 target=sm_89 block-m=64 block-n=64 block-k=16 threads=256 vector-width=4 stages=1" | FileCheck %s --check-prefix=TF32

func.func @batch_matmul(%lhs: memref<?x65x17xf32>,
                        %rhs: memref<?x17x67xf32>,
                        %out: memref<?x65x67xf32>) {
  linalg.batch_matmul
      ins(%lhs, %rhs : memref<?x65x17xf32>, memref<?x17x67xf32>)
      outs(%out : memref<?x65x67xf32>)
  return
}

// SHARED-LABEL: func.func @batch_matmul
// SHARED: %[[BATCH:.*]] = memref.dim %{{.*}}, %c0
// SHARED: gpu.launch blocks{{.*}}%[[BATCH]]) threads
// SHARED: nvgpu.device_async_copy {{.*}} : memref<?x65x17xf32> to memref<2x64x16xf32, #gpu.address_space<workgroup>>
// SHARED: nvgpu.device_async_copy {{.*}} : memref<?x17x67xf32> to memref<2x16x64xf32, #gpu.address_space<workgroup>>
// SHARED: nvgpu.device_async_wait
// SHARED: memref.store {{.*}} : memref<?x65x67xf32>
// SHARED-NOT: linalg.batch_matmul

// TF32-LABEL: func.func @batch_matmul
// TF32: gpu.launch blocks
// TF32: nvgpu.ldmatrix
// TF32: nvgpu.mma.sync
// TF32-SAME: mmaShape = [16, 8, 8]
// TF32-SAME: tf32Enabled
// TF32: memref.store {{.*}} : memref<?x65x67xf32>
// TF32-NOT: linalg.batch_matmul
