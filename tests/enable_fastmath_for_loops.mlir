// RUN: tutorial-opt %s -enable-fastmath-for-loops | FileCheck %s

// reassoc+contract is what LLVM's loop vectorizer requires before it will
// reorder a floating-point reduction; see lib/EnableFastMathForLoops.h for
// why -mcpu=native / -ffast-math alone cannot substitute for this.

func.func @reduce(%a: f32, %b: f32, %c: f32) -> f32 {
  %m = arith.mulf %a, %b : f32
  %s = arith.addf %c, %m : f32
  return %s : f32
}

// CHECK-LABEL: func.func @reduce
// CHECK: arith.mulf %{{.*}}, %{{.*}} fastmath<reassoc,contract> : f32
// CHECK: arith.addf %{{.*}}, %{{.*}} fastmath<reassoc,contract> : f32

// Ops already inside a loop nest (the case that actually matters: matmul
// lowered to scf.for by the --bufferization-to-llvm-generic fallback) get
// the same treatment.
func.func @reduce_loop(%a: memref<8xf32>, %b: memref<8xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %zero = arith.constant 0.0 : f32
  %result = scf.for %i = %c0 to %c8 step %c1 iter_args(%acc = %zero) -> f32 {
    %av = memref.load %a[%i] : memref<8xf32>
    %bv = memref.load %b[%i] : memref<8xf32>
    %prod = arith.mulf %av, %bv : f32
    %next = arith.addf %acc, %prod : f32
    scf.yield %next : f32
  }
  return %result : f32
}

// CHECK-LABEL: func.func @reduce_loop
// CHECK: arith.mulf %{{.*}}, %{{.*}} fastmath<reassoc,contract> : f32
// CHECK: arith.addf %{{.*}}, %{{.*}} fastmath<reassoc,contract> : f32
