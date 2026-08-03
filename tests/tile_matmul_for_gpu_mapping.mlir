// RUN: tutorial-opt %s -tile-matmul-for-gpu="block-m=8 block-n=32" | mlir-opt --gpu-map-parallel-loops="mapping-policy=innermost-first" --convert-parallel-loops-to-gpu --canonicalize | FileCheck %s

func.func @mapped_matmul(%lhs: memref<32x17xf32>,
                         %rhs: memref<17x64xf32>,
                         %out: memref<32x64xf32>) {
  linalg.matmul ins(%lhs, %rhs : memref<32x17xf32>, memref<17x64xf32>)
                outs(%out : memref<32x64xf32>)
  return
}

// CHECK-LABEL: func.func @mapped_matmul
// CHECK: gpu.launch blocks(%{{.*}}, %{{.*}}, %{{.*}}) in (%{{.*}} = %c2, %{{.*}} = %c4, %{{.*}} = %c1) threads(%{{.*}}, %{{.*}}, %{{.*}}) in (%{{.*}} = %c32, %{{.*}} = %c8, %{{.*}} = %c1)
// CHECK: scf.if
// CHECK: scf.for
// CHECK-NOT: scf.parallel
// CHECK-NOT: linalg.matmul
