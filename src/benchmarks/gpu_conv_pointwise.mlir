func.func @gpu_conv_pointwise(%input: memref<1x32x112x112xf32>,
                              %filter: memref<64x32x1x1xf32>,
                              %output: memref<1x64x112x112xf32>) {
  linalg.conv_2d_nchw_fchw
      {dilations = dense<1> : vector<2xi64>,
       strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : memref<1x32x112x112xf32>, memref<64x32x1x1xf32>)
      outs(%output : memref<1x64x112x112xf32>)
  return
}
