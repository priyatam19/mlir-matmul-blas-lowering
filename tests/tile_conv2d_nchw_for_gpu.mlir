// RUN: tutorial-opt %s -tile-conv2d-nchw-for-gpu="threads=256" | FileCheck %s

func.func @static_conv(%input: memref<1x3x7x7xf32>,
                       %filter: memref<2x3x3x3xf32>,
                       %output: memref<1x2x5x5xf32>) {
  linalg.conv_2d_nchw_fchw
      {dilations = dense<1> : vector<2xi64>,
       strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : memref<1x3x7x7xf32>, memref<2x3x3x3xf32>)
      outs(%output : memref<1x2x5x5xf32>)
  return
}

// CHECK-LABEL: func.func @static_conv
// CHECK: %[[N:.*]] = memref.dim %{{.*}}, %c0
// CHECK: %[[F:.*]] = memref.dim %{{.*}}, %c1
// CHECK: %[[OH:.*]] = memref.dim %{{.*}}, %c2
// CHECK: %[[OW:.*]] = memref.dim %{{.*}}, %c3
// CHECK: %[[TOTAL0:.*]] = arith.muli %[[N]], %[[F]]
// CHECK: %[[TOTAL1:.*]] = arith.muli %[[TOTAL0]], %[[OH]]
// CHECK: %[[TOTAL:.*]] = arith.muli %[[TOTAL1]], %[[OW]]
// CHECK: scf.parallel (%[[BLOCK:.*]]) = {{.*}} to (%[[TOTAL]]) step (%c256)
// CHECK: scf.parallel (%[[THREAD:.*]]) = {{.*}} to (%c256) step (%c1)
// CHECK: %[[LINEAR:.*]] = arith.addi %[[BLOCK]], %[[THREAD]]
// CHECK: arith.cmpi ult, %[[LINEAR]], %[[TOTAL]]
// CHECK: scf.if
// CHECK: arith.remui
// CHECK: arith.divui
// CHECK: %[[INITIAL:.*]] = memref.load
// CHECK: %[[CLOOP:.*]] = scf.for
// CHECK-SAME: iter_args(%{{.*}} = %[[INITIAL]]) -> (f32)
// CHECK: scf.for
// CHECK: scf.for
// CHECK: memref.load
// CHECK: memref.load
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: memref.store %[[CLOOP]]
// CHECK-NOT: linalg.conv_2d_nchw_fchw

func.func @dynamic_conv(%input: memref<?x?x?x?xf32>,
                        %filter: memref<?x?x?x?xf32>,
                        %output: memref<?x?x?x?xf32>) {
  linalg.conv_2d_nchw_fchw
      {dilations = dense<1> : vector<2xi64>,
       strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : memref<?x?x?x?xf32>, memref<?x?x?x?xf32>)
      outs(%output : memref<?x?x?x?xf32>)
  return
}

// CHECK-LABEL: func.func @dynamic_conv
// CHECK: scf.parallel
// CHECK: scf.parallel
// CHECK: scf.if
// CHECK-NOT: linalg.conv_2d_nchw_fchw

func.func @strided_conv(
    %input: memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>>,
    %filter: memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>>,
    %output: memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>>) {
  linalg.conv_2d_nchw_fchw
      {dilations = dense<1> : vector<2xi64>,
       strides = dense<1> : vector<2xi64>}
      ins(%input, %filter
          : memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>>,
            memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>>)
      outs(%output
           : memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>>)
  return
}

// CHECK-LABEL: func.func @strided_conv
// CHECK: scf.parallel
// CHECK-NOT: linalg.conv_2d_nchw_fchw

func.func @irregular_conv(%input: memref<1x2x9x10xf32>,
                          %filter: memref<3x2x3x2xf32>,
                          %output: memref<1x3x3x5xf32>) {
  linalg.conv_2d_nchw_fchw
      {dilations = dense<[2, 1]> : vector<2xi64>,
       strides = dense<2> : vector<2xi64>}
      ins(%input, %filter : memref<1x2x9x10xf32>, memref<3x2x3x2xf32>)
      outs(%output : memref<1x3x3x5xf32>)
  return
}

// CHECK-LABEL: func.func @irregular_conv
// CHECK-DAG: %[[S2:.*]] = arith.constant 2 : index
// CHECK: arith.muli {{.*}}, %[[S2]]
// CHECK-NOT: linalg.conv_2d_nchw_fchw

func.func @unsupported_f64(%input: memref<1x3x7x7xf64>,
                           %filter: memref<2x3x3x3xf64>,
                           %output: memref<1x2x5x5xf64>) {
  linalg.conv_2d_nchw_fchw
      {dilations = dense<1> : vector<2xi64>,
       strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : memref<1x3x7x7xf64>, memref<2x3x3x3xf64>)
      outs(%output : memref<1x2x5x5xf64>)
  return
}

// CHECK-LABEL: func.func @unsupported_f64
// CHECK: linalg.conv_2d_nchw_fchw
