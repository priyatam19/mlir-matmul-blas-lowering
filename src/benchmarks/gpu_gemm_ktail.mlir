func.func @gpu_gemm_ktail(%lhs: memref<1024x1031xf32>,
                          %rhs: memref<1031x1024xf32>,
                          %output: memref<1024x1024xf32>) {
  linalg.matmul ins(%lhs, %rhs : memref<1024x1031xf32>, memref<1031x1024xf32>)
                outs(%output : memref<1024x1024xf32>)
  return
}
