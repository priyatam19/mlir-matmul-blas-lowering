func.func @gpu_gemm_512(%a: memref<512x256xf32>,
                        %b: memref<256x512xf32>,
                        %c: memref<512x512xf32>) {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c512 = arith.constant 512 : index
  scf.for %m = %c0 to %c512 step %c16 {
    scf.for %n = %c0 to %c512 step %c16 {
      %a_tile = memref.subview %a[%m, 0] [16, 256] [1, 1]
          : memref<512x256xf32> to memref<16x256xf32, strided<[256, 1], offset: ?>>
      %b_tile = memref.subview %b[0, %n] [256, 16] [1, 1]
          : memref<256x512xf32> to memref<256x16xf32, strided<[512, 1], offset: ?>>
      %c_tile = memref.subview %c[%m, %n] [16, 16] [1, 1]
          : memref<512x512xf32> to memref<16x16xf32, strided<[512, 1], offset: ?>>
      linalg.matmul
          ins(%a_tile, %b_tile
              : memref<16x256xf32, strided<[256, 1], offset: ?>>,
                memref<256x16xf32, strided<[512, 1], offset: ?>>)
          outs(%c_tile : memref<16x16xf32, strided<[512, 1], offset: ?>>)
    }
  }
  return
}
