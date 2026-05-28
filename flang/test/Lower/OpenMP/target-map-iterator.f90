! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %s | FileCheck %s

! Tests for the iterator modifier on target map clauses.

subroutine target_map_iterator_simple()
  integer, parameter :: n = 16
  integer :: a(n)
  integer :: i

  !$omp target map(iterator(i = 1:n), tofrom: a(i))
    a(1) = 10
  !$omp end target
end subroutine

! CHECK-LABEL: func.func @_QPtarget_map_iterator_simple()
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtarget_map_iterator_simpleEa"}
! CHECK: %[[IT:.*]] = omp.iterator(%[[IV:.*]]: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[IV_I32:.*]] = fir.convert %[[IV]] : (index) -> i32
! CHECK:   %[[IV_I64:.*]] = fir.convert %[[IV_I32]] : (i32) -> i64
! CHECK:   %[[IV_IDX:.*]] = fir.convert %[[IV_I64]] : (i64) -> index
! CHECK:   %[[LB:.*]] = arith.subi %[[IV_IDX]], %{{.*}} : index
! CHECK:   %[[BOUNDS:.*]] = omp.map.bounds lower_bound(%[[LB]] : index) upper_bound(%[[LB]] : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index)
! CHECK:   %[[MAP:.*]] = omp.map.info var_ptr(%[[A]]#0 : !fir.ref<!fir.array<16xi32>>, !fir.array<16xi32>) map_clauses(tofrom) capture(ByRef) bounds(%[[BOUNDS]]) -> !llvm.ptr {name = ""}
! CHECK:   omp.yield(%[[MAP]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: omp.target map_iterated(%[[IT]] : !omp.iterated<!llvm.ptr>) map_iterated_captures(%[[A]]#0 -> %[[ARG:.*]] : !fir.ref<!fir.array<16xi32>>) {
! CHECK:   %[[TARGET_A:.*]]:2 = hlfir.declare %[[ARG]](%{{.*}}) {uniq_name = "_QFtarget_map_iterator_simpleEa"}
! CHECK:   hlfir.designate %[[TARGET_A]]#0

subroutine target_map_iterator_multi_obj()
  integer, parameter :: n = 16
  integer :: a(n), b(n)
  integer :: i

  !$omp target map(iterator(i = 1:n), to: a(i), b(i))
    a(1) = b(1)
  !$omp end target
end subroutine

! CHECK-LABEL: func.func @_QPtarget_map_iterator_multi_obj()
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtarget_map_iterator_multi_objEa"}
! CHECK: %[[B:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtarget_map_iterator_multi_objEb"}
! CHECK: %[[IT_A:.*]] = omp.iterator(%{{.*}}: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[MAP_A:.*]] = omp.map.info var_ptr(%[[A]]#0 : !fir.ref<!fir.array<16xi32>>, !fir.array<16xi32>) map_clauses(to) capture(ByRef) bounds({{.*}}) -> !llvm.ptr {name = ""}
! CHECK:   omp.yield(%[[MAP_A]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: %[[IT_B:.*]] = omp.iterator(%{{.*}}: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[MAP_B:.*]] = omp.map.info var_ptr(%[[B]]#0 : !fir.ref<!fir.array<16xi32>>, !fir.array<16xi32>) map_clauses(to) capture(ByRef) bounds({{.*}}) -> !llvm.ptr {name = ""}
! CHECK:   omp.yield(%[[MAP_B]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: omp.target map_iterated(%[[IT_A]], %[[IT_B]] : !omp.iterated<!llvm.ptr>, !omp.iterated<!llvm.ptr>) map_iterated_captures(%[[A]]#0 -> %[[ARG_A:.*]], %[[B]]#0 -> %[[ARG_B:.*]] : !fir.ref<!fir.array<16xi32>>, !fir.ref<!fir.array<16xi32>>) {
! CHECK:   hlfir.declare %[[ARG_A]]
! CHECK:   hlfir.declare %[[ARG_B]]

subroutine target_map_iterator_mixed()
  integer, parameter :: n = 16
  integer :: a(n)
  integer :: i

  !$omp target map(iterator(i = 2:n:2), tofrom: a(1), a(i))
    a(1) = 10
  !$omp end target
end subroutine

! CHECK-LABEL: func.func @_QPtarget_map_iterator_mixed()
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtarget_map_iterator_mixedEa"}
! CHECK: %[[MAP_PLAIN:.*]] = omp.map.info var_ptr(%[[A]]#1 : !fir.ref<!fir.array<16xi32>>, !fir.array<16xi32>) map_clauses(tofrom) capture(ByRef) bounds({{.*}}) -> !fir.ref<!fir.array<16xi32>> {name = "a(1)"}
! CHECK: %[[IT:.*]] = omp.iterator(%{{.*}}: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[MAP_IT:.*]] = omp.map.info var_ptr(%[[A]]#0 : !fir.ref<!fir.array<16xi32>>, !fir.array<16xi32>) map_clauses(tofrom) capture(ByRef) bounds({{.*}}) -> !llvm.ptr {name = ""}
! CHECK:   omp.yield(%[[MAP_IT]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: omp.target map_iterated(%[[IT]] : !omp.iterated<!llvm.ptr>) map_entries(%[[MAP_PLAIN]] -> %[[MAP_ARG:.*]] : !fir.ref<!fir.array<16xi32>>) map_iterated_captures(%[[A]]#0 -> %[[CAPTURE_ARG:.*]] : !fir.ref<!fir.array<16xi32>>) {
! CHECK:   hlfir.declare %[[MAP_ARG]]
! CHECK:   %[[CAPTURE_A:.*]]:2 = hlfir.declare %[[CAPTURE_ARG]](%{{.*}}) {uniq_name = "_QFtarget_map_iterator_mixedEa"}
! CHECK:   hlfir.designate %[[CAPTURE_A]]#0

subroutine target_map_iterator_duplicate_capture()
  integer, parameter :: n = 16
  integer :: a(n)
  integer :: i

  !$omp target map(iterator(i = 1:n-1), tofrom: a(i), a(i+1))
    a(1) = 10
  !$omp end target
end subroutine

! CHECK-LABEL: func.func @_QPtarget_map_iterator_duplicate_capture()
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtarget_map_iterator_duplicate_captureEa"}
! CHECK: %[[IT_A:.*]] = omp.iterator(%{{.*}}: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[MAP_A:.*]] = omp.map.info var_ptr(%[[A]]#0 : !fir.ref<!fir.array<16xi32>>, !fir.array<16xi32>) map_clauses(tofrom) capture(ByRef) bounds({{.*}}) -> !llvm.ptr {name = ""}
! CHECK:   omp.yield(%[[MAP_A]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: %[[IT_A_NEXT:.*]] = omp.iterator(%{{.*}}: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[MAP_A_NEXT:.*]] = omp.map.info var_ptr(%[[A]]#0 : !fir.ref<!fir.array<16xi32>>, !fir.array<16xi32>) map_clauses(tofrom) capture(ByRef) bounds({{.*}}) -> !llvm.ptr {name = ""}
! CHECK:   omp.yield(%[[MAP_A_NEXT]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: omp.target map_iterated(%[[IT_A]], %[[IT_A_NEXT]] : !omp.iterated<!llvm.ptr>, !omp.iterated<!llvm.ptr>) map_iterated_captures(%[[A]]#0 -> %[[ARG:.*]] : !fir.ref<!fir.array<16xi32>>) {
! CHECK:   hlfir.declare %[[ARG]]
