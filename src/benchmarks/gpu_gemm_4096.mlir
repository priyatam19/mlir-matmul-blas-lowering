func.func @gpu_gemm_4096(%lhs: memref<4096x4096xf32>,
                         %rhs: memref<4096x4096xf32>,
                         %output: memref<4096x4096xf32>) {
  linalg.matmul ins(%lhs, %rhs : memref<4096x4096xf32>, memref<4096x4096xf32>)
                outs(%output : memref<4096x4096xf32>)
  return
}
