func.func @gpu_conv_resnet_stem(%input: memref<1x3x230x230xf32>,
                                %filter: memref<64x3x7x7xf32>,
                                %output: memref<1x64x112x112xf32>) {
  linalg.conv_2d_nchw_fchw
      {dilations = dense<1> : vector<2xi64>,
       strides = dense<2> : vector<2xi64>}
      ins(%input, %filter : memref<1x3x230x230xf32>, memref<64x3x7x7xf32>)
      outs(%output : memref<1x64x112x112xf32>)
  return
}
