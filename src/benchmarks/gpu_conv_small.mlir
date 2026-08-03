func.func @gpu_conv_small(%input: memref<4x1x30x30xf32>,
                          %filter: memref<32x1x3x3xf32>,
                          %output: memref<4x32x28x28xf32>) {
  linalg.conv_2d_nchw_fchw
      {dilations = dense<1> : vector<2xi64>,
       strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : memref<4x1x30x30xf32>, memref<32x1x3x3xf32>)
      outs(%output : memref<4x32x28x28xf32>)
  return
}
