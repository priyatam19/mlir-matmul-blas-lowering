func.func @gpu_conv_irregular(%input: memref<2x7x35x37xf32>,
                              %filter: memref<13x7x3x5xf32>,
                              %output: memref<2x13x16x33xf32>) {
  linalg.conv_2d_nchw_fchw
      {dilations = dense<[2, 1]> : vector<2xi64>,
       strides = dense<[2, 1]> : vector<2xi64>}
      ins(%input, %filter : memref<2x7x35x37xf32>, memref<13x7x3x5xf32>)
      outs(%output : memref<2x13x16x33xf32>)
  return
}
