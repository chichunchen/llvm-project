// RUN: fir-opt --omp-function-filtering %s | FileCheck %s

module attributes {omp.is_target_device = true} {
  // CHECK-LABEL: func.func @map_iterated_target
  // CHECK-SAME: (%[[ARRAY:.*]]: !fir.ref<!fir.array<8xi32>>, %[[UB:.*]]: index)
  func.func @map_iterated_target(%array: !fir.ref<!fir.array<8xi32>>,
                                 %ub: index) {
    // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : i64
    // CHECK-NEXT: %[[SHAPE:.*]] = fir.shape %[[C0]] : (i64) -> !fir.shape<1>
    // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
    // CHECK-DAG: %[[C8:.*]] = arith.constant 8 : index
    // CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[ARRAY]](%[[SHAPE]])
    %c8 = arith.constant 8 : index
    %shape = fir.shape %c8 : (index) -> !fir.shape<1>
    %decl:2 = hlfir.declare %array(%shape) {uniq_name = "array"} :
        (!fir.ref<!fir.array<8xi32>>, !fir.shape<1>) ->
        (!fir.ref<!fir.array<8xi32>>, !fir.ref<!fir.array<8xi32>>)
    %c1 = arith.constant 1 : index
    %iter = omp.iterator(%iv: index) = (%c1 to %ub step %c1) {
      %offset = arith.subi %iv, %c1 : index
      %bounds = omp.map.bounds lower_bound(%offset : index)
                                   upper_bound(%offset : index)
                                   extent(%c8 : index)
                                   stride(%c1 : index)
                                   start_idx(%c1 : index)
      %map = omp.map.info var_ptr(%decl#0 : !fir.ref<!fir.array<8xi32>>,
                                  !fir.array<8xi32>)
          map_clauses(tofrom) capture(ByRef) bounds(%bounds) -> !llvm.ptr
      omp.yield(%map : !llvm.ptr)
    } -> !omp.iterated<!llvm.ptr>

    // CHECK-NEXT: %[[ITER:.*]] = omp.iterator(%[[IV:.*]]: index) =
    // CHECK-SAME: (%[[C1]] to %[[UB]] step %[[C1]])
    // CHECK: %[[MAP:.*]] = omp.map.info var_ptr(%[[DECL]]#0
    // CHECK: omp.yield(%[[MAP]] : !llvm.ptr)
    // CHECK: omp.target map_iterated(%[[ITER]] : !omp.iterated<!llvm.ptr>)
    // CHECK-SAME: map_iterated_captures(%[[DECL]]#0 -> %{{.*}}
    omp.target map_iterated(%iter : !omp.iterated<!llvm.ptr>)
        map_iterated_captures(%decl#0 -> %arg0 :
                              !fir.ref<!fir.array<8xi32>>) {
      %c8_0 = arith.constant 8 : index
      %shape_0 = fir.shape %c8_0 : (index) -> !fir.shape<1>
      %target_decl:2 = hlfir.declare %arg0(%shape_0) {uniq_name = "array"} :
          (!fir.ref<!fir.array<8xi32>>, !fir.shape<1>) ->
          (!fir.ref<!fir.array<8xi32>>, !fir.ref<!fir.array<8xi32>>)
      omp.terminator
    }
    return
  }
}
