! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %s | FileCheck %s

! Tests for the iterator modifier on the to/from motion clauses across all
! directives that support them: target update, target enter data, and
! target exit data.

!===============================================================================
! target update
!===============================================================================

subroutine target_update_to_simple()
  integer, parameter :: n = 16
  integer :: a(n)
  integer :: i

  !$omp target update to(iterator(i = 1:n): a(i))
end subroutine

! CHECK-LABEL: func.func @_QPtarget_update_to_simple()
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtarget_update_to_simpleEa"}
! CHECK: %[[IT:.*]] = omp.iterator(%[[IV:.*]]: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[IV_I32:.*]] = fir.convert %[[IV]] : (index) -> i32
! CHECK:   %[[IV_I64:.*]] = fir.convert %[[IV_I32]] : (i32) -> i64
! CHECK:   %[[IV_IDX:.*]] = fir.convert %[[IV_I64]] : (i64) -> index
! CHECK:   %[[LB:.*]] = arith.subi %[[IV_IDX]], %{{.*}} : index
! CHECK:   %[[BOUNDS:.*]] = omp.map.bounds lower_bound(%[[LB]] : index) upper_bound(%[[LB]] : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index)
! CHECK:   %[[MAP:.*]] = omp.map.info var_ptr(%[[A]]#0 : !fir.ref<!fir.array<16xi32>>, !fir.array<16xi32>) map_clauses(to) capture(ByRef) bounds(%[[BOUNDS]]) -> !llvm.ptr {name = ""}
! CHECK:   omp.yield(%[[MAP]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: omp.target_update map_entries(%[[IT]] : !omp.iterated<!llvm.ptr>)

subroutine target_update_from_simple()
  integer, parameter :: n = 16
  integer :: a(n)
  integer :: i

  !$omp target update from(iterator(i = 1:n): a(i))
end subroutine

! CHECK-LABEL: func.func @_QPtarget_update_from_simple()
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtarget_update_from_simpleEa"}
! CHECK: %[[IT:.*]] = omp.iterator(%[[IV:.*]]: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[BOUNDS:.*]] = omp.map.bounds lower_bound(%{{.*}} : index) upper_bound(%{{.*}} : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index)
! CHECK:   %[[MAP:.*]] = omp.map.info var_ptr(%[[A]]#0 : !fir.ref<!fir.array<16xi32>>, !fir.array<16xi32>) map_clauses(from) capture(ByRef) bounds(%[[BOUNDS]]) -> !llvm.ptr {name = ""}
! CHECK:   omp.yield(%[[MAP]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: omp.target_update map_entries(%[[IT]] : !omp.iterated<!llvm.ptr>)

subroutine target_update_2d()
  integer, parameter :: n = 4, m = 6
  integer :: a(n, m)
  integer :: i, j

  !$omp target update to(iterator(i = 1:n, j = 1:m): a(i, j))
end subroutine

! CHECK-LABEL: func.func @_QPtarget_update_2d()
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtarget_update_2dEa"}
! CHECK: %[[IT:.*]] = omp.iterator(%[[IV0:.*]]: index, %[[IV1:.*]]: index) = ({{.*}} to {{.*}} step {{.*}}, {{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[IV0_I32:.*]] = fir.convert %[[IV0]] : (index) -> i32
! CHECK:   %[[IV1_I32:.*]] = fir.convert %[[IV1]] : (index) -> i32
! CHECK:   %[[IV0_I64:.*]] = fir.convert %[[IV0_I32]] : (i32) -> i64
! CHECK:   %[[IV1_I64:.*]] = fir.convert %[[IV1_I32]] : (i32) -> i64
! CHECK:   %[[BOUNDS0:.*]] = omp.map.bounds lower_bound(%{{.*}} : index) upper_bound(%{{.*}} : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index)
! CHECK:   %[[BOUNDS1:.*]] = omp.map.bounds lower_bound(%{{.*}} : index) upper_bound(%{{.*}} : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index)
! CHECK:   %[[MAP:.*]] = omp.map.info var_ptr(%[[A]]#0 : !fir.ref<!fir.array<4x6xi32>>, !fir.array<4x6xi32>) map_clauses(to) capture(ByRef) bounds(%[[BOUNDS0]], %[[BOUNDS1]]) -> !llvm.ptr {name = ""}
! CHECK:   omp.yield(%[[MAP]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: omp.target_update map_entries(%[[IT]] : !omp.iterated<!llvm.ptr>)

subroutine target_update_step()
  integer, parameter :: n = 16
  integer :: a(n)
  integer :: i

  !$omp target update to(iterator(i = 1:n:2): a(i))
end subroutine

! CHECK-LABEL: func.func @_QPtarget_update_step()
! CHECK: %[[C1_I32:.*]] = arith.constant 1 : i32
! CHECK: %[[C16_I32:.*]] = arith.constant 16 : i32
! CHECK: %[[LB:.*]] = fir.convert %[[C1_I32]] : (i32) -> index
! CHECK: %[[UB:.*]] = fir.convert %[[C16_I32]] : (i32) -> index
! CHECK: %[[C2_I32:.*]] = arith.constant 2 : i32
! CHECK: %[[STEP:.*]] = fir.convert %[[C2_I32]] : (i32) -> index
! CHECK: %[[IT:.*]] = omp.iterator(%[[IV:.*]]: index) = (%[[LB]] to %[[UB]] step %[[STEP]]) {
! CHECK:   %[[BOUNDS:.*]] = omp.map.bounds lower_bound(%{{.*}} : index) upper_bound(%{{.*}} : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index)
! CHECK:   %[[MAP:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<!fir.array<16xi32>>, !fir.array<16xi32>) map_clauses(to) capture(ByRef) bounds(%[[BOUNDS]]) -> !llvm.ptr {name = ""}
! CHECK:   omp.yield(%[[MAP]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: omp.target_update map_entries(%[[IT]] : !omp.iterated<!llvm.ptr>)

subroutine target_update_negative_step()
  integer, parameter :: n = 16
  integer :: a(n)
  integer :: i

  !$omp target update to(iterator(i = n:1:-1): a(i))
end subroutine

! CHECK-LABEL: func.func @_QPtarget_update_negative_step()
! CHECK: %[[C16_I32:.*]] = arith.constant 16 : i32
! CHECK: %[[C1_I32:.*]] = arith.constant 1 : i32
! CHECK: %[[LB:.*]] = fir.convert %[[C16_I32]] : (i32) -> index
! CHECK: %[[UB:.*]] = fir.convert %[[C1_I32]] : (i32) -> index
! CHECK: %[[CM1_I32:.*]] = arith.constant -1 : i32
! CHECK: %[[STEP:.*]] = fir.convert %[[CM1_I32]] : (i32) -> index
! CHECK: %[[IT:.*]] = omp.iterator(%[[IV:.*]]: index) = (%[[LB]] to %[[UB]] step %[[STEP]]) {
! CHECK:   %[[BOUNDS:.*]] = omp.map.bounds lower_bound(%{{.*}} : index) upper_bound(%{{.*}} : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index)
! CHECK:   %[[MAP:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<!fir.array<16xi32>>, !fir.array<16xi32>) map_clauses(to) capture(ByRef) bounds(%[[BOUNDS]]) -> !llvm.ptr {name = ""}
! CHECK:   omp.yield(%[[MAP]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: omp.target_update map_entries(%[[IT]] : !omp.iterated<!llvm.ptr>)

subroutine target_update_multi_obj()
  integer, parameter :: n = 16
  integer :: a(n), b(n)
  integer :: i

  !$omp target update to(iterator(i = 1:n): a(i), b(i))
end subroutine

! CHECK-LABEL: func.func @_QPtarget_update_multi_obj()
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtarget_update_multi_objEa"}
! CHECK: %[[B:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtarget_update_multi_objEb"}
! CHECK: %[[IT1:.*]] = omp.iterator(%{{.*}}: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[BOUNDS1:.*]] = omp.map.bounds lower_bound(%{{.*}} : index) upper_bound(%{{.*}} : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index)
! CHECK:   %[[MAP1:.*]] = omp.map.info var_ptr(%[[A]]#0 : !fir.ref<!fir.array<16xi32>>, !fir.array<16xi32>) map_clauses(to) capture(ByRef) bounds(%[[BOUNDS1]]) -> !llvm.ptr {name = ""}
! CHECK:   omp.yield(%[[MAP1]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: %[[IT2:.*]] = omp.iterator(%{{.*}}: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[BOUNDS2:.*]] = omp.map.bounds lower_bound(%{{.*}} : index) upper_bound(%{{.*}} : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index)
! CHECK:   %[[MAP2:.*]] = omp.map.info var_ptr(%[[B]]#0 : !fir.ref<!fir.array<16xi32>>, !fir.array<16xi32>) map_clauses(to) capture(ByRef) bounds(%[[BOUNDS2]]) -> !llvm.ptr {name = ""}
! CHECK:   omp.yield(%[[MAP2]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: omp.target_update map_entries(%[[IT1]], %[[IT2]] : !omp.iterated<!llvm.ptr>, !omp.iterated<!llvm.ptr>)

subroutine target_update_mixed_same_clause()
  integer, parameter :: n = 16
  integer :: a(n)
  integer :: i

  !$omp target update to(iterator(i = 2:n:2): a(1), a(i))
end subroutine

! CHECK-LABEL: func.func @_QPtarget_update_mixed_same_clause()
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtarget_update_mixed_same_clauseEa"}
! CHECK: %[[MAP_PLAIN:.*]] = omp.map.info var_ptr(%[[A]]#1 : !fir.ref<!fir.array<16xi32>>, !fir.array<16xi32>) map_clauses(to) capture(ByRef) bounds({{.*}}) -> !fir.ref<!fir.array<16xi32>> {name = "a(1)"}
! CHECK: %[[IT:.*]] = omp.iterator(%[[IV:.*]]: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[BOUNDS_IT:.*]] = omp.map.bounds lower_bound(%{{.*}} : index) upper_bound(%{{.*}} : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index)
! CHECK:   %[[MAP_IT:.*]] = omp.map.info var_ptr(%[[A]]#0 : !fir.ref<!fir.array<16xi32>>, !fir.array<16xi32>) map_clauses(to) capture(ByRef) bounds(%[[BOUNDS_IT]]) -> !llvm.ptr {name = ""}
! CHECK:   omp.yield(%[[MAP_IT]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: omp.target_update map_entries(%[[MAP_PLAIN]], %[[IT]] : !fir.ref<!fir.array<16xi32>>, !omp.iterated<!llvm.ptr>)

subroutine target_update_multi_clause()
  integer, parameter :: n = 8
  integer :: a(n), b(n)
  integer :: i, j

  !$omp target update to(iterator(i = 1:n): a(i)) &
  !$omp&              from(iterator(j = 1:n:2): b(j))
end subroutine

! CHECK-LABEL: func.func @_QPtarget_update_multi_clause()
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtarget_update_multi_clauseEa"}
! CHECK: %[[B:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtarget_update_multi_clauseEb"}
! CHECK: %[[IT1:.*]] = omp.iterator(%{{.*}}: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[BOUNDS1:.*]] = omp.map.bounds lower_bound(%{{.*}} : index) upper_bound(%{{.*}} : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index)
! CHECK:   %[[MAP1:.*]] = omp.map.info var_ptr(%[[A]]#0 : !fir.ref<!fir.array<8xi32>>, !fir.array<8xi32>) map_clauses(to) capture(ByRef) bounds(%[[BOUNDS1]]) -> !llvm.ptr {name = ""}
! CHECK:   omp.yield(%[[MAP1]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: %[[IT2:.*]] = omp.iterator(%{{.*}}: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[BOUNDS2:.*]] = omp.map.bounds lower_bound(%{{.*}} : index) upper_bound(%{{.*}} : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index)
! CHECK:   %[[MAP2:.*]] = omp.map.info var_ptr(%[[B]]#0 : !fir.ref<!fir.array<8xi32>>, !fir.array<8xi32>) map_clauses(from) capture(ByRef) bounds(%[[BOUNDS2]]) -> !llvm.ptr {name = ""}
! CHECK:   omp.yield(%[[MAP2]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: omp.target_update map_entries(%[[IT1]], %[[IT2]] : !omp.iterated<!llvm.ptr>, !omp.iterated<!llvm.ptr>)

subroutine target_update_mixed_clauses()
  integer, parameter :: n = 16
  integer :: a(n), b(n)
  integer :: i

  !$omp target update to(iterator(i = 1:n): a(i)) from(b)
end subroutine

! CHECK-LABEL: func.func @_QPtarget_update_mixed_clauses()
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtarget_update_mixed_clausesEa"}
! CHECK: %[[B:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtarget_update_mixed_clausesEb"}
! CHECK: %[[IT:.*]] = omp.iterator(%{{.*}}: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[BOUNDS_IT:.*]] = omp.map.bounds lower_bound(%{{.*}} : index) upper_bound(%{{.*}} : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index)
! CHECK:   %[[MAP_IT:.*]] = omp.map.info var_ptr(%[[A]]#0 : !fir.ref<!fir.array<16xi32>>, !fir.array<16xi32>) map_clauses(to) capture(ByRef) bounds(%[[BOUNDS_IT]]) -> !llvm.ptr {name = ""}
! CHECK:   omp.yield(%[[MAP_IT]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: %[[MAP_B:.*]] = omp.map.info var_ptr(%[[B]]#1 : !fir.ref<!fir.array<16xi32>>, !fir.array<16xi32>) map_clauses(from) capture(ByRef) bounds({{.*}}) -> !fir.ref<!fir.array<16xi32>> {name = "b"}
! CHECK: omp.target_update map_entries(%[[MAP_B]], %[[IT]] : !fir.ref<!fir.array<16xi32>>, !omp.iterated<!llvm.ptr>)

!===============================================================================
! target enter data
!===============================================================================

subroutine target_enter_data_simple()
  integer, parameter :: n = 16
  integer :: a(n)
  integer :: i

  !$omp target enter data map(iterator(i = 1:n), to: a(i))
end subroutine

! CHECK-LABEL: func.func @_QPtarget_enter_data_simple()
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtarget_enter_data_simpleEa"}
! CHECK: %[[IT:.*]] = omp.iterator(%[[IV:.*]]: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[IV_I32:.*]] = fir.convert %[[IV]] : (index) -> i32
! CHECK:   %[[IV_I64:.*]] = fir.convert %[[IV_I32]] : (i32) -> i64
! CHECK:   %[[IV_IDX:.*]] = fir.convert %[[IV_I64]] : (i64) -> index
! CHECK:   %[[LB:.*]] = arith.subi %[[IV_IDX]], %{{.*}} : index
! CHECK:   %[[BOUNDS:.*]] = omp.map.bounds lower_bound(%[[LB]] : index) upper_bound(%[[LB]] : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index)
! CHECK:   %[[MAP:.*]] = omp.map.info var_ptr(%[[A]]#0 : !fir.ref<!fir.array<16xi32>>, !fir.array<16xi32>) map_clauses(to) capture(ByRef) bounds(%[[BOUNDS]]) -> !llvm.ptr {name = ""}
! CHECK:   omp.yield(%[[MAP]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: omp.target_enter_data map_entries(%[[IT]] : !omp.iterated<!llvm.ptr>)

! Expression-based subscript using multiple iterator variables: a((i-1)*m+j)
! maps a 2D logical iteration space onto a 1D array.
subroutine target_enter_data_expr_subscript()
  integer, parameter :: m = 4
  integer, parameter :: n = m * m
  integer :: a(n)
  integer :: i, j

  !$omp target enter data map(iterator(i = 1:m, j = 1:m), to: a((i-1)*m+j))
end subroutine

! CHECK-LABEL: func.func @_QPtarget_enter_data_expr_subscript()
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtarget_enter_data_expr_subscriptEa"}
! CHECK: %[[IT:.*]] = omp.iterator(%[[IV0:.*]]: index, %[[IV1:.*]]: index) = ({{.*}} to {{.*}} step {{.*}}, {{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[IV0_I32:.*]] = fir.convert %[[IV0]] : (index) -> i32
! CHECK:   %[[IV1_I32:.*]] = fir.convert %[[IV1]] : (index) -> i32
! CHECK:   %[[C1_I32:.*]] = arith.constant 1 : i32
! CHECK:   %[[SUB:.*]] = arith.subi %[[IV0_I32]], %[[C1_I32]] : i32
! CHECK:   %[[NOREASSOC:.*]] = fir.no_reassoc %[[SUB]] : i32
! CHECK:   %[[MUL:.*]] = arith.muli %{{.*}}, %[[NOREASSOC]] : i32
! CHECK:   %[[ADD:.*]] = arith.addi %[[MUL]], %[[IV1_I32]] : i32
! CHECK:   %[[IDX:.*]] = fir.convert %[[ADD]] : (i32) -> i64
! CHECK:   %[[BOUNDS:.*]] = omp.map.bounds lower_bound(%{{.*}} : index) upper_bound(%{{.*}} : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index)
! CHECK:   %[[MAP:.*]] = omp.map.info var_ptr(%[[A]]#0 : !fir.ref<!fir.array<16xi32>>, !fir.array<16xi32>) map_clauses(to) capture(ByRef) bounds(%[[BOUNDS]]) -> !llvm.ptr {name = ""}
! CHECK:   omp.yield(%[[MAP]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: omp.target_enter_data map_entries(%[[IT]] : !omp.iterated<!llvm.ptr>)

!===============================================================================
! target exit data
!===============================================================================

subroutine target_exit_data_simple()
  integer, parameter :: n = 16
  integer :: a(n)
  integer :: i

  !$omp target exit data map(iterator(i = 1:n), from: a(i))
end subroutine

! CHECK-LABEL: func.func @_QPtarget_exit_data_simple()
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtarget_exit_data_simpleEa"}
! CHECK: %[[IT:.*]] = omp.iterator(%[[IV:.*]]: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[IV_I32:.*]] = fir.convert %[[IV]] : (index) -> i32
! CHECK:   %[[IV_I64:.*]] = fir.convert %[[IV_I32]] : (i32) -> i64
! CHECK:   %[[IV_IDX:.*]] = fir.convert %[[IV_I64]] : (i64) -> index
! CHECK:   %[[LB:.*]] = arith.subi %[[IV_IDX]], %{{.*}} : index
! CHECK:   %[[BOUNDS:.*]] = omp.map.bounds lower_bound(%[[LB]] : index) upper_bound(%[[LB]] : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index)
! CHECK:   %[[MAP:.*]] = omp.map.info var_ptr(%[[A]]#0 : !fir.ref<!fir.array<16xi32>>, !fir.array<16xi32>) map_clauses(from) capture(ByRef) bounds(%[[BOUNDS]]) -> !llvm.ptr {name = ""}
! CHECK:   omp.yield(%[[MAP]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: omp.target_exit_data map_entries(%[[IT]] : !omp.iterated<!llvm.ptr>)

! Multiple objects with negative step, producing separate iterators.
subroutine target_exit_data_multi_obj()
  integer, parameter :: n = 16
  integer :: a(n), b(n)
  integer :: i

  !$omp target exit data map(iterator(i = n:1:-1), from: a(i), b(i))
end subroutine

! CHECK-LABEL: func.func @_QPtarget_exit_data_multi_obj()
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtarget_exit_data_multi_objEa"}
! CHECK: %[[B:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtarget_exit_data_multi_objEb"}
! CHECK: %[[C16_I32:.*]] = arith.constant 16 : i32
! CHECK: %[[C1_I32:.*]] = arith.constant 1 : i32
! CHECK: %[[LB:.*]] = fir.convert %[[C16_I32]] : (i32) -> index
! CHECK: %[[UB:.*]] = fir.convert %[[C1_I32]] : (i32) -> index
! CHECK: %[[CM1_I32:.*]] = arith.constant -1 : i32
! CHECK: %[[STEP:.*]] = fir.convert %[[CM1_I32]] : (i32) -> index
! CHECK: %[[IT1:.*]] = omp.iterator(%{{.*}}: index) = (%[[LB]] to %[[UB]] step %[[STEP]]) {
! CHECK:   %[[BOUNDS1:.*]] = omp.map.bounds lower_bound(%{{.*}} : index) upper_bound(%{{.*}} : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index)
! CHECK:   %[[MAP1:.*]] = omp.map.info var_ptr(%[[A]]#0 : !fir.ref<!fir.array<16xi32>>, !fir.array<16xi32>) map_clauses(from) capture(ByRef) bounds(%[[BOUNDS1]]) -> !llvm.ptr {name = ""}
! CHECK:   omp.yield(%[[MAP1]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: %[[IT2:.*]] = omp.iterator(%{{.*}}: index) = (%[[LB]] to %[[UB]] step %[[STEP]]) {
! CHECK:   %[[BOUNDS2:.*]] = omp.map.bounds lower_bound(%{{.*}} : index) upper_bound(%{{.*}} : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index)
! CHECK:   %[[MAP2:.*]] = omp.map.info var_ptr(%[[B]]#0 : !fir.ref<!fir.array<16xi32>>, !fir.array<16xi32>) map_clauses(from) capture(ByRef) bounds(%[[BOUNDS2]]) -> !llvm.ptr {name = ""}
! CHECK:   omp.yield(%[[MAP2]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: omp.target_exit_data map_entries(%[[IT1]], %[[IT2]] : !omp.iterated<!llvm.ptr>, !omp.iterated<!llvm.ptr>)
