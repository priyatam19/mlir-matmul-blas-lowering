// RUN: tutorial-opt %s -tile-batch-matmul-for-gpu="block-m=8 block-n=32" | mlir-opt --gpu-map-parallel-loops="mapping-policy=innermost-first" --convert-parallel-loops-to-gpu --canonicalize | FileCheck %s

func.func @mapped_batch_matmul(%lhs: memref<2x33x17xf32>,
                               %rhs: memref<2x17x65xf32>,
                               %out: memref<2x33x65xf32>) {
  linalg.batch_matmul
      ins(%lhs, %rhs : memref<2x33x17xf32>, memref<2x17x65xf32>)
      outs(%out : memref<2x33x65xf32>)
  return
}

// CHECK-LABEL: func.func @mapped_batch_matmul
// CHECK: gpu.launch blocks(%{{.*}}, %{{.*}}, %{{.*}}) in (%{{.*}} = %c3, %{{.*}} = %c5, %{{.*}} = %c2) threads(%{{.*}}, %{{.*}}, %{{.*}}) in (%{{.*}} = %c32, %{{.*}} = %c8, %{{.*}} = %c1)
// CHECK: scf.if
// CHECK: scf.for
// CHECK-NOT: scf.parallel
// CHECK-NOT: linalg.batch_matmul
