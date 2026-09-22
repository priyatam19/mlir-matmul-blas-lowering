func.func @gpu_gemm_wide(%lhs: memref<256x1024xf32>,
                         %rhs: memref<1024x4096xf32>,
                         %output: memref<256x4096xf32>) {
  linalg.matmul ins(%lhs, %rhs : memref<256x1024xf32>, memref<1024x4096xf32>)
                outs(%output : memref<256x4096xf32>)
  return
}
