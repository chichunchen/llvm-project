// RUN: fir-opt --lower-workdistribute %s | FileCheck %s

// Keep a lower-workdistribute test for target map iterator captures because the
// pass splits and hoists target regions. That path must preserve the dialect
// block-argument order where map_iterated_captures are between ordinary
// map_entries and private arguments.

module attributes {llvm.target_triple = "x86_64-unknown-linux-gnu", omp.is_gpu = false, omp.is_target_device = false} {
// CHECK-LABEL: func.func @target_map_iterated_capture_workdistribute
// CHECK-SAME: %{{arg[0-9]+}}: !fir.ref<index>, %[[CAPTURE:arg[0-9]+]]: !fir.ref<index>)
func.func @target_map_iterated_capture_workdistribute(%lb : index, %ub : index,
                                                      %step : index,
                                                      %addr : !fir.ref<index>,
                                                      %capture : !fir.ref<index>) {
  %lb_ref = fir.alloca index {bindc_name = "lb"}
  fir.store %lb to %lb_ref : !fir.ref<index>
  %ub_ref = fir.alloca index {bindc_name = "ub"}
  fir.store %ub to %ub_ref : !fir.ref<index>
  %step_ref = fir.alloca index {bindc_name = "step"}
  fir.store %step to %step_ref : !fir.ref<index>

  %lb_map = omp.map.info var_ptr(%lb_ref : !fir.ref<index>, index) map_clauses(to) capture(ByRef) -> !fir.ref<index> {name = "lb"}
  %ub_map = omp.map.info var_ptr(%ub_ref : !fir.ref<index>, index) map_clauses(to) capture(ByRef) -> !fir.ref<index> {name = "ub"}
  %step_map = omp.map.info var_ptr(%step_ref : !fir.ref<index>, index) map_clauses(to) capture(ByRef) -> !fir.ref<index> {name = "step"}
  %addr_map = omp.map.info var_ptr(%addr : !fir.ref<index>, index) map_clauses(tofrom) capture(ByRef) -> !fir.ref<index> {name = "addr"}

  %it = omp.iterator(%iv: index) = (%lb to %ub step %step) {
    %map = omp.map.info var_ptr(%addr : !fir.ref<index>, index) map_clauses(tofrom) capture(ByRef) -> !fir.ref<index> {name = "iter_addr"}
    omp.yield(%map : !fir.ref<index>)
  } -> !omp.iterated<!fir.ref<index>>

  omp.target map_iterated(%it : !omp.iterated<!fir.ref<index>>) map_entries(%lb_map -> %ARG0, %ub_map -> %ARG1, %step_map -> %ARG2, %addr_map -> %ARG3 : !fir.ref<index>, !fir.ref<index>, !fir.ref<index>, !fir.ref<index>) map_iterated_captures(%capture -> %CAPTURE : !fir.ref<index>) {
    %lb_val = fir.load %ARG0 : !fir.ref<index>
    %ub_val = fir.load %ARG1 : !fir.ref<index>
    %step_val = fir.load %ARG2 : !fir.ref<index>
    %capture_val = fir.load %CAPTURE : !fir.ref<index>
    %sum = arith.addi %ub_val, %capture_val : index
    omp.teams {
      omp.workdistribute {
        fir.do_loop %iv = %lb_val to %ub_val step %step_val unordered {
          fir.store %sum to %ARG3 : !fir.ref<index>
        }
        omp.terminator
      }
      omp.terminator
    }
    omp.terminator
  }
  return
}
}

// CHECK: omp.target_data map_entries
// CHECK: %[[CAPTURE_VAL:.*]] = fir.load %[[CAPTURE]] : !fir.ref<index>
// CHECK: omp.target map_iterated(%{{.*}} : !omp.iterated<!fir.ref<index>>) host_eval
// CHECK-SAME: map_iterated_captures(%[[CAPTURE]] -> %[[CAPTURE_ARG:.*]] : !fir.ref<index>)
// CHECK: %[[SUM:.*]] = arith.addi %{{.*}}, %{{.*}} : index
// CHECK: omp.loop_nest
// CHECK: fir.store %[[SUM]] to %{{.*}} : !fir.ref<index>
