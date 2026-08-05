// RUN: tutorial-opt %s -convert-batch-matmul-to-cublas | FileCheck %s

#lhs_transpose = affine_map<(b, m, n, k) -> (b, k, m)>
#rhs_default = affine_map<(b, m, n, k) -> (b, k, n)>
#out_default = affine_map<(b, m, n, k) -> (b, m, n)>

// CHECK: func.func private @tutorial_cublas_sgemm_strided_batched_f32(memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>, memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>, memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>)

func.func @static_batch_matmul(%lhs: memref<2x33x17xf32>,
                               %rhs: memref<2x17x65xf32>,
                               %out: memref<2x33x65xf32>) {
  linalg.batch_matmul
      ins(%lhs, %rhs : memref<2x33x17xf32>, memref<2x17x65xf32>)
      outs(%out : memref<2x33x65xf32>)
  return
}

// CHECK-LABEL: func.func @static_batch_matmul
// CHECK: %[[LHS:.*]] = memref.cast %{{.*}} : memref<2x33x17xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
// CHECK: %[[RHS:.*]] = memref.cast %{{.*}} : memref<2x17x65xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
// CHECK: %[[OUT:.*]] = memref.cast %{{.*}} : memref<2x33x65xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
// CHECK: call @tutorial_cublas_sgemm_strided_batched_f32(%[[LHS]], %[[RHS]], %[[OUT]])
// CHECK-NOT: linalg.batch_matmul

func.func @dynamic_strided(
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

// CHECK-LABEL: func.func @dynamic_strided
// CHECK: call @tutorial_cublas_sgemm_strided_batched_f32
// CHECK-NOT: linalg.batch_matmul

func.func @unsupported_inner_stride(
    %lhs: memref<2x4x8xf32, strided<[64, 16, 2]>>,
    %rhs: memref<2x8x4xf32>, %out: memref<2x4x4xf32>) {
  linalg.batch_matmul
      ins(%lhs, %rhs : memref<2x4x8xf32, strided<[64, 16, 2]>>,
                        memref<2x8x4xf32>)
      outs(%out : memref<2x4x4xf32>)
  return
}

// CHECK-LABEL: func.func @unsupported_inner_stride
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
