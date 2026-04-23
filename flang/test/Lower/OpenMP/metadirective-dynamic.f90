! Test lowering of OpenMP metadirective with dynamic user conditions.
! Verifies that non-constant user conditions are lowered to fir.if chains.

! RUN: %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=52 %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPtest_dynamic_condition(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<!fir.logical<4>>
! CHECK:         %[[DECL:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:         %[[LOAD:.*]] = fir.load %[[DECL]]#0
! CHECK:         %[[COND:.*]] = fir.convert %[[LOAD]] : (!fir.logical<4>) -> i1
! CHECK:         fir.if %[[COND]] {
! CHECK:           omp.barrier
! CHECK:         } else {
! CHECK:         }
! CHECK:         return
subroutine test_dynamic_condition(flag)
  logical :: flag
  ! Dynamic condition: should produce fir.if with omp.barrier in the then
  ! branch and nothing in the else branch.
  !$omp metadirective &
  !$omp & when(user={condition(flag)}: barrier) &
  !$omp & otherwise(nothing)
end subroutine
