// RUN: tutorial-opt %s -convert-conv2d-nchw-to-cudnn | FileCheck %s

// CHECK: func.func private @tutorial_cudnn_conv2d_nchw_f32(memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>>, index, index, index, index)

func.func @static_conv(%input: memref<2x7x35x37xf32>,
                       %filter: memref<13x7x3x5xf32>,
                       %output: memref<2x13x16x33xf32>) {
  linalg.conv_2d_nchw_fchw
      {dilations = dense<[2, 1]> : vector<2xi64>,
       strides = dense<[2, 1]> : vector<2xi64>}
      ins(%input, %filter : memref<2x7x35x37xf32>, memref<13x7x3x5xf32>)
      outs(%output : memref<2x13x16x33xf32>)
  return
}

// CHECK-LABEL: func.func @static_conv
// CHECK-DAG: %[[INPUT:.*]] = memref.cast %{{.*}} : memref<2x7x35x37xf32> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>>
// CHECK-DAG: %[[FILTER:.*]] = memref.cast %{{.*}} : memref<13x7x3x5xf32> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>>
// CHECK-DAG: %[[OUTPUT:.*]] = memref.cast %{{.*}} : memref<2x13x16x33xf32> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>>
// CHECK-DAG: %[[STRIDE_H:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[STRIDE_W:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[DILATION_H:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[DILATION_W:.*]] = arith.constant 1 : index
// CHECK: call @tutorial_cudnn_conv2d_nchw_f32(%[[INPUT]], %[[FILTER]], %[[OUTPUT]], %[[STRIDE_H]], %[[STRIDE_W]], %[[DILATION_H]], %[[DILATION_W]])
// CHECK-NOT: linalg.conv_2d_nchw_fchw

func.func @dynamic_strided(
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

// CHECK-LABEL: func.func @dynamic_strided
// CHECK: call @tutorial_cudnn_conv2d_nchw_f32
// CHECK-NOT: linalg.conv_2d_nchw_fchw

func.func @unsupported_filter_layout(
    %input: memref<1x3x7x7xf32>,
    %filter: memref<2x3x3x3xf32, strided<[81, 27, 9, 2]>>,
    %output: memref<1x2x5x5xf32>) {
  linalg.conv_2d_nchw_fchw
      {dilations = dense<1> : vector<2xi64>,
       strides = dense<1> : vector<2xi64>}
      ins(%input, %filter
          : memref<1x3x7x7xf32>,
            memref<2x3x3x3xf32, strided<[81, 27, 9, 2]>>)
      outs(%output : memref<1x2x5x5xf32>)
  return
}

// CHECK-LABEL: func.func @unsupported_filter_layout
// CHECK: linalg.conv_2d_nchw_fchw

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
