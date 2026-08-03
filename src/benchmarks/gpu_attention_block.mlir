#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
module {
  func.func @attention_block(%arg0: tensor<12x128x64xf32>, %arg1: tensor<12x128x64xf32>, %arg2: tensor<12x128x64xf32>) -> tensor<12x128x64xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %c0_i64 = arith.constant 0 : i64
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 1.250000e-01 : f32
    %0 = tensor.empty() : tensor<12x64x128xf32>
    %transposed = linalg.transpose ins(%arg1 : tensor<12x128x64xf32>) outs(%0 : tensor<12x64x128xf32>) permutation = [0, 2, 1]
    %1 = tensor.empty() : tensor<12x128x128xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<12x128x128xf32>) -> tensor<12x128x128xf32>
    %3 = linalg.batch_matmul ins(%arg0, %transposed : tensor<12x128x64xf32>, tensor<12x64x128xf32>) outs(%2 : tensor<12x128x128xf32>) -> tensor<12x128x128xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3 : tensor<12x128x128xf32>) outs(%1 : tensor<12x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %19 = arith.mulf %in, %cst_1 : f32
      linalg.yield %19 : f32
    } -> tensor<12x128x128xf32>
    %5 = tensor.empty() : tensor<12x128xi64>
    %6 = linalg.fill ins(%c0_i64 : i64) outs(%5 : tensor<12x128xi64>) -> tensor<12x128xi64>
    %7 = tensor.empty() : tensor<12x128xf32>
    %8 = linalg.fill ins(%cst_0 : f32) outs(%7 : tensor<12x128xf32>) -> tensor<12x128xf32>
    %9:2 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%4 : tensor<12x128x128xf32>) outs(%8, %6 : tensor<12x128xf32>, tensor<12x128xi64>) {
    ^bb0(%in: f32, %out: f32, %out_2: i64):
      %19 = linalg.index 2 : index
      %20 = arith.index_cast %19 : index to i64
      %21 = arith.maximumf %in, %out : f32
      %22 = arith.cmpf ogt, %in, %out : f32
      %23 = arith.select %22, %20, %out_2 : i64
      linalg.yield %21, %23 : f32, i64
    } -> (tensor<12x128xf32>, tensor<12x128xi64>)
    %expanded = tensor.expand_shape %9#0 [[0], [1, 2]] output_shape [12, 128, 1] : tensor<12x128xf32> into tensor<12x128x1xf32>
    %10 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4, %expanded : tensor<12x128x128xf32>, tensor<12x128x1xf32>) outs(%1 : tensor<12x128x128xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %19 = arith.subf %in, %in_2 : f32
      linalg.yield %19 : f32
    } -> tensor<12x128x128xf32>
    %11 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%10 : tensor<12x128x128xf32>) outs(%1 : tensor<12x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %19 = math.exp %in : f32
      linalg.yield %19 : f32
    } -> tensor<12x128x128xf32>
    %12 = tensor.empty() : tensor<12x128x1xf32>
    %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<12x128x1xf32>) -> tensor<12x128x1xf32>
    %14 = linalg.generic {indexing_maps = [#map, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%11 : tensor<12x128x128xf32>) outs(%13 : tensor<12x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %19 = arith.addf %in, %out : f32
      linalg.yield %19 : f32
    } -> tensor<12x128x1xf32>
    %15 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%11, %14 : tensor<12x128x128xf32>, tensor<12x128x1xf32>) outs(%1 : tensor<12x128x128xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %19 = arith.divf %in, %in_2 : f32
      linalg.yield %19 : f32
    } -> tensor<12x128x128xf32>
    %16 = tensor.empty() : tensor<12x128x64xf32>
    %17 = linalg.fill ins(%cst : f32) outs(%16 : tensor<12x128x64xf32>) -> tensor<12x128x64xf32>
    %18 = linalg.batch_matmul ins(%15, %arg2 : tensor<12x128x128xf32>, tensor<12x128x64xf32>) outs(%17 : tensor<12x128x64xf32>) -> tensor<12x128x64xf32>
    return %18 : tensor<12x128x64xf32>
  }
}
