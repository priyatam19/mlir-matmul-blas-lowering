func.func @gpu_gemm_mobilenet(%a: memref<3136x32xf32>,
                              %b: memref<32x64xf32>,
                              %c: memref<3136x64xf32>) {
  linalg.matmul ins(%a, %b : memref<3136x32xf32>, memref<32x64xf32>)
                outs(%c : memref<3136x64xf32>)
  return
}
