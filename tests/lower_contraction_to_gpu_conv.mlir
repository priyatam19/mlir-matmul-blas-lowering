// RUN: tutorial-opt %s -lower-contraction-to-gpu="strategy=shared-fp32 target=sm_89 block-m=64 block-n=64 block-k=16 threads=256 vector-width=4 stages=1" | FileCheck %s --check-prefix=SHARED
// RUN: tutorial-opt %s -lower-contraction-to-gpu="strategy=tensorcore-tf32 target=sm_89 block-m=64 block-n=64 block-k=16 threads=256 vector-width=4 stages=1" | FileCheck %s --check-prefix=TF32

func.func @implicit_gemm_conv(%input: memref<?x?x?x?xf32>,
                              %filter: memref<?x?x?x?xf32>,
                              %output: memref<?x?x?x?xf32>) {
  linalg.conv_2d_nchw_fchw
      {dilations = dense<[2, 1]> : vector<2xi64>,
       strides = dense<[2, 1]> : vector<2xi64>}
      ins(%input, %filter : memref<?x?x?x?xf32>, memref<?x?x?x?xf32>)
      outs(%output : memref<?x?x?x?xf32>)
  return
}

// SHARED-LABEL: func.func @implicit_gemm_conv
// SHARED: gpu.launch {{.*}} workgroup(%[[FILTER:.*]] : memref<64x16xf32, #gpu.address_space<workgroup>>, %[[PATCH:.*]] : memref<16x65xf32, #gpu.address_space<workgroup>>)
// SHARED: memref.load {{.*}} : memref<?x?x?x?xf32>
// SHARED: memref.store {{.*}}, %[[FILTER]]
// SHARED: arith.muli
// SHARED: vector.load {{.*}} : memref<?x?x?x?xf32>, vector<4xf32>
// SHARED: vector.store {{.*}}, %[[PATCH]]
// SHARED: memref.load {{.*}} : memref<?x?x?x?xf32>
// SHARED: memref.store {{.*}}, %[[PATCH]]
// SHARED: gpu.barrier
// SHARED: arith.mulf
// SHARED: arith.addf
// SHARED: memref.store {{.*}} : memref<?x?x?x?xf32>
// SHARED-NOT: linalg.conv_2d_nchw_fchw

// TF32-LABEL: func.func @implicit_gemm_conv
// TF32: gpu.launch
// TF32: gpu.barrier
// TF32: nvgpu.ldmatrix
// TF32: nvgpu.mma.sync
// TF32-SAME: mmaShape = [16, 8, 8]
// TF32-SAME: tf32Enabled
// TF32: memref.store {{.*}} : memref<?x?x?x?xf32>
// TF32-NOT: linalg.conv_2d_nchw_fchw
