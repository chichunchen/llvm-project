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

! Dynamic condition with target variant: verifies host_eval processing
! works when the evaluation is a metadirective rather than an OpenMPConstruct.
! CHECK-LABEL: func.func @_QPtest_dynamic_target(
! CHECK:         fir.if
! CHECK:           omp.target
! CHECK:             omp.teams
! CHECK:               omp.parallel
! CHECK:                 omp.distribute
! CHECK:                   omp.wsloop
! CHECK:                     omp.simd
! CHECK:                       omp.loop_nest
! CHECK:         } else {
! CHECK:           omp.parallel
! CHECK:             omp.wsloop
! CHECK:               omp.simd
! CHECK:                 omp.loop_nest
subroutine test_dynamic_target(flag, a, n)
  logical, intent(in) :: flag
  real, intent(inout) :: a(:)
  integer, intent(in) :: n
  !$omp metadirective &
  !$omp & when(user={condition(flag)}: target teams distribute parallel do simd) &
  !$omp & otherwise(parallel do simd)
  do i = 1, n
    a(i) = a(i) + 1.0
  end do
end subroutine
