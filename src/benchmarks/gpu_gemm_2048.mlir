func.func @gpu_gemm_2048(%a: memref<2048x2048xf32>,
                         %b: memref<2048x2048xf32>,
                         %c: memref<2048x2048xf32>) {
  linalg.matmul ins(%a, %b : memref<2048x2048xf32>, memref<2048x2048xf32>)
                outs(%c : memref<2048x2048xf32>)
  return
}
