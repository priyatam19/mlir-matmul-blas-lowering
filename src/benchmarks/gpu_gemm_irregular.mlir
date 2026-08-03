func.func @gpu_gemm_irregular(%a: memref<513x257xf32>,
                              %b: memref<257x509xf32>,
                              %c: memref<513x509xf32>) {
  linalg.matmul ins(%a, %b : memref<513x257xf32>, memref<257x509xf32>)
                outs(%c : memref<513x509xf32>)
  return
}
