func.func @gpu_gemm_tall(%lhs: memref<4096x1024xf32>,
                         %rhs: memref<1024x256xf32>,
                         %output: memref<4096x256xf32>) {
  linalg.matmul ins(%lhs, %rhs : memref<4096x1024xf32>, memref<1024x256xf32>)
                outs(%output : memref<4096x256xf32>)
  return
}
