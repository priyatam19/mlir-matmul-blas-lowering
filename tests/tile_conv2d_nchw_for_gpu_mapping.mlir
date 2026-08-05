// RUN: tutorial-opt %s -tile-conv2d-nchw-for-gpu="threads=256" | mlir-opt --gpu-map-parallel-loops="mapping-policy=innermost-first" --convert-parallel-loops-to-gpu --canonicalize | FileCheck %s

func.func @mapped_conv(%input: memref<1x3x7x7xf32>,
                       %filter: memref<2x3x3x3xf32>,
                       %output: memref<1x2x5x5xf32>) {
  linalg.conv_2d_nchw_fchw
      {dilations = dense<1> : vector<2xi64>,
       strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : memref<1x3x7x7xf32>, memref<2x3x3x3xf32>)
      outs(%output : memref<1x2x5x5xf32>)
  return
}

// CHECK-LABEL: func.func @mapped_conv
// CHECK: gpu.launch blocks(%{{.*}}, %{{.*}}, %{{.*}}) in (%{{.*}} = %c1, %{{.*}} = %c1, %{{.*}} = %c1) threads(%{{.*}}, %{{.*}}, %{{.*}}) in (%{{.*}} = %c256, %{{.*}} = %c1, %{{.*}} = %c1)
// CHECK: scf.if
// CHECK-COUNT-3: scf.for
// CHECK-NOT: scf.parallel
// CHECK-NOT: linalg.conv_2d_nchw_fchw
