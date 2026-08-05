// RUN: tutorial-opt %s -lower-contraction-to-gpu="strategy=autotuned target=sm_89" | FileCheck %s

func.func @autotuned_bmm(%lhs: memref<?x?x?xf32>,
                         %rhs: memref<?x?x?xf32>,
                         %out: memref<?x?x?xf32>) {
  linalg.batch_matmul
      ins(%lhs, %rhs : memref<?x?x?xf32>, memref<?x?x?xf32>)
      outs(%out : memref<?x?x?xf32>)
  return
}

// CHECK-LABEL: func.func @autotuned_bmm
// CHECK: %[[BMM_CANDIDATE:.*]] = call @tutorial_autotune_begin
// CHECK: scf.index_switch %[[BMM_CANDIDATE]]
// CHECK: case 0 {
// CHECK: scf.parallel
// CHECK: case 1 {
// CHECK: gpu.launch
// CHECK: default {
// CHECK: gpu.launch
// CHECK: call @tutorial_autotune_end
// CHECK-NOT: linalg.batch_matmul

func.func @autotuned_conv(%input: memref<?x?x?x?xf32>,
                          %filter: memref<?x?x?x?xf32>,
                          %output: memref<?x?x?x?xf32>) {
  linalg.conv_2d_nchw_fchw
      {dilations = dense<1> : vector<2xi64>,
       strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : memref<?x?x?x?xf32>, memref<?x?x?x?xf32>)
      outs(%output : memref<?x?x?x?xf32>)
  return
}

// CHECK-LABEL: func.func @autotuned_conv
// CHECK: %[[CONV_CANDIDATE:.*]] = call @tutorial_autotune_begin
// CHECK: scf.index_switch %[[CONV_CANDIDATE]]
// CHECK: case 0 {
// CHECK: gpu.launch
// CHECK: case 6 {
// CHECK: gpu.launch
// CHECK: default {
// CHECK: gpu.launch
// CHECK: call @tutorial_autotune_end
// CHECK-NOT: linalg.conv_2d_nchw_fchw
