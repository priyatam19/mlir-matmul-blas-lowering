#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  func.func @residual_conv_block(%arg0: tensor<1x16x32x32xf32>, %arg1: tensor<16x16x3x3xf32>, %arg2: tensor<16xf32>, %arg3: tensor<16x16x3x3xf32>, %arg4: tensor<16xf32>) -> tensor<1x16x32x32xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %padded = tensor.pad %arg0 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg5: index, %arg6: index, %arg7: index, %arg8: index):
      tensor.yield %cst : f32
    } : tensor<1x16x32x32xf32> to tensor<1x16x34x34xf32>
    %0 = tensor.empty() : tensor<1x16x32x32xf32>
    %broadcasted = linalg.broadcast ins(%arg2 : tensor<16xf32>) outs(%0 : tensor<1x16x32x32xf32>) dimensions = [0, 2, 3]
    %1 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded, %arg1 : tensor<1x16x34x34xf32>, tensor<16x16x3x3xf32>) outs(%broadcasted : tensor<1x16x32x32xf32>) -> tensor<1x16x32x32xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1 : tensor<1x16x32x32xf32>) outs(%0 : tensor<1x16x32x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %6 = arith.cmpf ugt, %in, %cst : f32
      %7 = arith.select %6, %in, %cst : f32
      linalg.yield %7 : f32
    } -> tensor<1x16x32x32xf32>
    %padded_0 = tensor.pad %2 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg5: index, %arg6: index, %arg7: index, %arg8: index):
      tensor.yield %cst : f32
    } : tensor<1x16x32x32xf32> to tensor<1x16x34x34xf32>
    %broadcasted_1 = linalg.broadcast ins(%arg4 : tensor<16xf32>) outs(%0 : tensor<1x16x32x32xf32>) dimensions = [0, 2, 3]
    %3 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_0, %arg3 : tensor<1x16x34x34xf32>, tensor<16x16x3x3xf32>) outs(%broadcasted_1 : tensor<1x16x32x32xf32>) -> tensor<1x16x32x32xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3, %arg0 : tensor<1x16x32x32xf32>, tensor<1x16x32x32xf32>) outs(%0 : tensor<1x16x32x32xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %6 = arith.addf %in, %in_2 : f32
      linalg.yield %6 : f32
    } -> tensor<1x16x32x32xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4 : tensor<1x16x32x32xf32>) outs(%0 : tensor<1x16x32x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %6 = arith.cmpf ugt, %in, %cst : f32
      %7 = arith.select %6, %in, %cst : f32
      linalg.yield %7 : f32
    } -> tensor<1x16x32x32xf32>
    return %5 : tensor<1x16x32x32xf32>
  }
}
