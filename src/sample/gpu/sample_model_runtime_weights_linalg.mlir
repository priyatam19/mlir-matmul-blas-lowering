#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @sample_model(%input: tensor<3x4xf32>, %input_shift: tensor<3x4xf32>, %weight_t: tensor<4x5xf32>, %bias: tensor<5xf32>) -> tensor<3x5xf32> {
    %zero = arith.constant 0.000000e+00 : f32
    %one = arith.constant 1.000000e+00 : f32
    %shifted_empty = tensor.empty() : tensor<3x4xf32>
    %shifted = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%input, %input_shift : tensor<3x4xf32>, tensor<3x4xf32>) outs(%shifted_empty : tensor<3x4xf32>) {
    ^bb0(%in: f32, %shift: f32, %out: f32):
      %sum = arith.addf %in, %shift : f32
      linalg.yield %sum : f32
    } -> tensor<3x4xf32>
    %matmul_empty = tensor.empty() : tensor<3x5xf32>
    %matmul_init = linalg.fill ins(%zero : f32) outs(%matmul_empty : tensor<3x5xf32>) -> tensor<3x5xf32>
    %matmul = linalg.matmul ins(%shifted, %weight_t : tensor<3x4xf32>, tensor<4x5xf32>) outs(%matmul_init : tensor<3x5xf32>) -> tensor<3x5xf32>
    %biased_empty = tensor.empty() : tensor<3x5xf32>
    %biased = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%matmul, %bias : tensor<3x5xf32>, tensor<5xf32>) outs(%biased_empty : tensor<3x5xf32>) {
    ^bb0(%in: f32, %bias_value: f32, %out: f32):
      %sum = arith.addf %in, %bias_value : f32
      linalg.yield %sum : f32
    } -> tensor<3x5xf32>
    %clamped_empty = tensor.empty() : tensor<3x5xf32>
    %clamped = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%biased : tensor<3x5xf32>) outs(%clamped_empty : tensor<3x5xf32>) {
    ^bb0(%in: f32, %out: f32):
      %is_lt_zero = arith.cmpf ult, %in, %zero : f32
      %lower = arith.select %is_lt_zero, %zero, %in : f32
      %is_gt_one = arith.cmpf ugt, %lower, %one : f32
      %upper = arith.select %is_gt_one, %one, %lower : f32
      linalg.yield %upper : f32
    } -> tensor<3x5xf32>
    return %clamped : tensor<3x5xf32>
  }
}
