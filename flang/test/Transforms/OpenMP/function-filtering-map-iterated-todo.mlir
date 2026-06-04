// RUN: not fir-opt --omp-function-filtering -o - %s 2>&1 | FileCheck %s

module attributes {omp.is_target_device = true} {
  func.func @target_map_iterated(%lb : index, %ub : index, %step : index,
                                 %addr : !fir.ref<i32>) {
    %it = omp.iterator(%iv: index) = (%lb to %ub step %step) {
      %map = omp.map.info var_ptr(%addr : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<i32>
      omp.yield(%map : !fir.ref<i32>)
    } -> !omp.iterated<!fir.ref<i32>>

    // CHECK: not yet implemented: Unhandled clause map_iterated in omp.target operation during function filtering
    omp.target map_iterated(%it : !omp.iterated<!fir.ref<i32>>) map_iterated_captures(%addr -> %arg0 : !fir.ref<i32>) {
      %0 = fir.load %arg0 : !fir.ref<i32>
      omp.terminator
    }
    return
  }
}
