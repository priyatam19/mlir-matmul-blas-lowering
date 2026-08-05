func.func @gpu_conv_resnet_block(%input: memref<1x64x58x58xf32>,
                                 %filter: memref<64x64x3x3xf32>,
                                 %output: memref<1x64x56x56xf32>) {
  linalg.conv_2d_nchw_fchw
      {dilations = dense<1> : vector<2xi64>,
       strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : memref<1x64x58x58xf32>, memref<64x64x3x3xf32>)
      outs(%output : memref<1x64x56x56xf32>)
  return
}
