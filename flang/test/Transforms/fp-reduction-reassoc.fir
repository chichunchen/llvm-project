// RUN: fir-opt --split-input-file --fp-reduction-reassoc %s | FileCheck %s

module {
  func.func @memory_reduction() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2.000000e+00 : f64
    %c3 = arith.constant 3.000000e+00 : f64
    %zero = arith.constant 0.000000e+00 : f64
    %q = fir.alloca f64
    fir.store %zero to %q : !fir.ref<f64>

    fir.do_loop %i = %c0 to %c4 step %c1 {
      %old = fir.load %q : !fir.ref<f64>
      %prod = arith.mulf %c2, %c3 fastmath<contract> : f64
      %sum = arith.addf %old, %prod fastmath<contract> : f64
      fir.store %sum to %q : !fir.ref<f64>
    }
    return
  }
}

// CHECK-LABEL: func.func @memory_reduction
// CHECK: fir.do_loop
// CHECK: %[[PROD:.*]] = arith.mulf %{{.*}}, %{{.*}} fastmath<reassoc,contract> : f64
// CHECK: %[[SUM:.*]] = arith.addf %{{.*}}, %[[PROD]] fastmath<reassoc,contract> : f64

// -----

module {
  func.func @iterarg_reduction() -> f64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2.000000e+00 : f64
    %c3 = arith.constant 3.000000e+00 : f64
    %zero = arith.constant 0.000000e+00 : f64

    %res = fir.do_loop %i = %c0 to %c4 step %c1 iter_args(%acc = %zero) -> (f64) {
      %prod = arith.mulf %c2, %c3 fastmath<contract> : f64
      %sum = arith.addf %acc, %prod fastmath<contract> : f64
      fir.result %sum : f64
    }
    return %res : f64
  }
}

// CHECK-LABEL: func.func @iterarg_reduction
// CHECK: fir.do_loop
// CHECK: %[[PROD:.*]] = arith.mulf %{{.*}}, %{{.*}} fastmath<reassoc,contract> : f64
// CHECK: %[[SUM:.*]] = arith.addf %{{.*}}, %[[PROD]] fastmath<reassoc,contract> : f64

// -----

module {
  func.func @not_a_reduction() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2.000000e+00 : f64
    %c3 = arith.constant 3.000000e+00 : f64
    %q = fir.alloca f64

    fir.do_loop %i = %c0 to %c4 step %c1 {
      %sum = arith.addf %c2, %c3 fastmath<contract> : f64
      fir.store %sum to %q : !fir.ref<f64>
    }
    return
  }
}

// CHECK-LABEL: func.func @not_a_reduction
// CHECK: fir.do_loop
// CHECK: arith.addf %{{.*}}, %{{.*}} fastmath<contract> : f64
// CHECK-NOT: fastmath<reassoc,contract>
