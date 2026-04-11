// RUN: fir-opt --fp-reduction-reassoc < %s | FileCheck %s

// -----

// Test Pattern 1: iter_arg-based reduction Q = Q + Z(k)*X(k)
// CHECK-LABEL: func.func @test_iterarg_dot_product
func.func @test_iterarg_dot_product(%arg0: !fir.ref<!fir.array<1001xf64>>,
                                     %arg1: !fir.ref<!fir.array<1001xf64>>) -> f64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c1001 = arith.constant 1001 : index
  %cst = arith.constant 0.000000e+00 : f64
  %result = fir.do_loop %iv = %c0 to %c1001 step %c1 iter_args(%q = %cst) -> (f64) {
    %addr_z = fir.coordinate_of %arg0, %iv : (!fir.ref<!fir.array<1001xf64>>, index) -> !fir.ref<f64>
    %z = fir.load %addr_z : !fir.ref<f64>
    %addr_x = fir.coordinate_of %arg1, %iv : (!fir.ref<!fir.array<1001xf64>>, index) -> !fir.ref<f64>
    %x = fir.load %addr_x : !fir.ref<f64>
    // CHECK: arith.mulf %{{.*}}, %{{.*}} fastmath<reassoc> : f64
    %prod = arith.mulf %z, %x : f64
    // CHECK: arith.addf %{{.*}}, %{{.*}} fastmath<reassoc> : f64
    %sum = arith.addf %q, %prod : f64
    fir.result %sum : f64
  }
  return %result : f64
}

// -----

// Test Pattern 1 with existing contract flag — reassoc is OR'd in
// CHECK-LABEL: func.func @test_iterarg_existing_flags
func.func @test_iterarg_existing_flags(%arg0: !fir.ref<!fir.array<100xf64>>,
                                        %arg1: !fir.ref<!fir.array<100xf64>>) -> f64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %cst = arith.constant 0.000000e+00 : f64
  %result = fir.do_loop %iv = %c0 to %c100 step %c1 iter_args(%q = %cst) -> (f64) {
    %addr_z = fir.coordinate_of %arg0, %iv : (!fir.ref<!fir.array<100xf64>>, index) -> !fir.ref<f64>
    %z = fir.load %addr_z : !fir.ref<f64>
    %addr_x = fir.coordinate_of %arg1, %iv : (!fir.ref<!fir.array<100xf64>>, index) -> !fir.ref<f64>
    %x = fir.load %addr_x : !fir.ref<f64>
    // CHECK: arith.mulf %{{.*}}, %{{.*}} fastmath<reassoc,contract> : f64
    %prod = arith.mulf %z, %x fastmath<contract> : f64
    // CHECK: arith.addf %{{.*}}, %{{.*}} fastmath<reassoc,contract> : f64
    %sum = arith.addf %q, %prod fastmath<contract> : f64
    fir.result %sum : f64
  }
  return %result : f64
}

// -----

// Test Pattern 2: memory-based reduction (load/add/store to same ref outside loop)
// CHECK-LABEL: func.func @test_memory_reduction
func.func @test_memory_reduction(%arg0: !fir.ref<!fir.array<1001xf64>>,
                                  %arg1: !fir.ref<!fir.array<1001xf64>>,
                                  %arg2: !fir.ref<f64>) {
  %c1 = arith.constant 1 : index
  %c1001 = arith.constant 1001 : index
  fir.do_loop %iv = %c1 to %c1001 step %c1 {
    %q = fir.load %arg2 : !fir.ref<f64>
    %addr_z = fir.coordinate_of %arg0, %iv : (!fir.ref<!fir.array<1001xf64>>, index) -> !fir.ref<f64>
    %z = fir.load %addr_z : !fir.ref<f64>
    %addr_x = fir.coordinate_of %arg1, %iv : (!fir.ref<!fir.array<1001xf64>>, index) -> !fir.ref<f64>
    %x = fir.load %addr_x : !fir.ref<f64>
    // CHECK: arith.mulf %{{.*}}, %{{.*}} fastmath<reassoc,contract> : f64
    %prod = arith.mulf %z, %x fastmath<contract> : f64
    // CHECK: arith.addf %{{.*}}, %{{.*}} fastmath<reassoc,contract> : f64
    %sum = arith.addf %q, %prod fastmath<contract> : f64
    fir.store %sum to %arg2 : !fir.ref<f64>
  }
  return
}

// -----

// Test: addf without mulf — only addf gets reassoc
// CHECK-LABEL: func.func @test_iterarg_add_only
func.func @test_iterarg_add_only(%arg0: !fir.ref<!fir.array<100xf64>>) -> f64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %cst = arith.constant 0.000000e+00 : f64
  %result = fir.do_loop %iv = %c0 to %c100 step %c1 iter_args(%q = %cst) -> (f64) {
    %addr = fir.coordinate_of %arg0, %iv : (!fir.ref<!fir.array<100xf64>>, index) -> !fir.ref<f64>
    %val = fir.load %addr : !fir.ref<f64>
    // CHECK: arith.addf %{{.*}}, %{{.*}} fastmath<reassoc> : f64
    %sum = arith.addf %q, %val : f64
    fir.result %sum : f64
  }
  return %result : f64
}

// -----

// Negative test: fir.no_reassoc guards the operand — no reassoc added
// CHECK-LABEL: func.func @test_no_reassoc_guard
func.func @test_no_reassoc_guard(%arg0: !fir.ref<!fir.array<100xf64>>,
                                  %arg1: !fir.ref<!fir.array<100xf64>>) -> f64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %cst = arith.constant 0.000000e+00 : f64
  %result = fir.do_loop %iv = %c0 to %c100 step %c1 iter_args(%q = %cst) -> (f64) {
    %addr_z = fir.coordinate_of %arg0, %iv : (!fir.ref<!fir.array<100xf64>>, index) -> !fir.ref<f64>
    %z = fir.load %addr_z : !fir.ref<f64>
    %addr_x = fir.coordinate_of %arg1, %iv : (!fir.ref<!fir.array<100xf64>>, index) -> !fir.ref<f64>
    %x = fir.load %addr_x : !fir.ref<f64>
    // CHECK: arith.mulf %{{.*}}, %{{.*}} : f64
    %prod = arith.mulf %z, %x : f64
    %guarded = fir.no_reassoc %prod : f64
    // CHECK: arith.addf %{{.*}}, %{{.*}} : f64
    %sum = arith.addf %q, %guarded : f64
    fir.result %sum : f64
  }
  return %result : f64
}

// -----

// Negative test: non-float iter_arg (integer) — should not be modified
// CHECK-LABEL: func.func @test_integer_iterarg_unchanged
func.func @test_integer_iterarg_unchanged() -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %result = fir.do_loop %iv = %c0 to %c100 step %c1 iter_args(%acc = %c0_i32) -> (i32) {
    // CHECK: arith.addi %{{.*}}, %{{.*}} : i32
    %sum = arith.addi %acc, %c1_i32 : i32
    fir.result %sum : i32
  }
  return %result : i32
}

// -----

// Negative test: mulf with multiple uses — only addf gets reassoc, mulf does not
// CHECK-LABEL: func.func @test_mulf_multiuse
func.func @test_mulf_multiuse(%arg0: !fir.ref<!fir.array<100xf64>>,
                               %arg1: !fir.ref<!fir.array<100xf64>>,
                               %arg2: !fir.ref<f64>) -> f64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %cst = arith.constant 0.000000e+00 : f64
  %result = fir.do_loop %iv = %c0 to %c100 step %c1 iter_args(%q = %cst) -> (f64) {
    %addr_z = fir.coordinate_of %arg0, %iv : (!fir.ref<!fir.array<100xf64>>, index) -> !fir.ref<f64>
    %z = fir.load %addr_z : !fir.ref<f64>
    %addr_x = fir.coordinate_of %arg1, %iv : (!fir.ref<!fir.array<100xf64>>, index) -> !fir.ref<f64>
    %x = fir.load %addr_x : !fir.ref<f64>
    // CHECK: arith.mulf %{{.*}}, %{{.*}} : f64
    %prod = arith.mulf %z, %x : f64
    fir.store %prod to %arg2 : !fir.ref<f64>
    // CHECK: arith.addf %{{.*}}, %{{.*}} fastmath<reassoc> : f64
    %sum = arith.addf %q, %prod : f64
    fir.result %sum : f64
  }
  return %result : f64
}
