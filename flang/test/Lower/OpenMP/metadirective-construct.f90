! Test metadirective with construct trait selectors.
! Verifies that enclosing OpenMP constructs are properly detected
! and matched against construct={...} context selectors.

! RUN: %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=52 %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPtest_construct_parallel()
! CHECK:         omp.parallel {
! CHECK:           omp.barrier
! CHECK:           omp.terminator
! CHECK:         }
subroutine test_construct_parallel()
  ! Inside parallel — construct={parallel} should match and emit barrier.
  !$omp parallel
  !$omp metadirective &
  !$omp & when(construct={parallel}: barrier) &
  !$omp & otherwise(nothing)
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPtest_construct_no_match()
! CHECK-NOT:     omp.parallel
! CHECK-NOT:     omp.barrier
! CHECK:         return
subroutine test_construct_no_match()
  ! Inside parallel — construct={target} should NOT match; fall to otherwise
  ! (nothing). Empty parallel is elided by lowering.
  !$omp parallel
  !$omp metadirective &
  !$omp & when(construct={target}: barrier) &
  !$omp & otherwise(nothing)
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPtest_construct_nested()
! CHECK:         omp.parallel {
! CHECK:           omp.wsloop
! CHECK:             omp.loop_nest
! CHECK:           omp.terminator
! CHECK:         }
! CHECK:         omp.parallel {
! CHECK:           omp.barrier
! CHECK:           omp.terminator
! CHECK:         }
subroutine test_construct_nested()
  integer :: i
  ! The parallel do is sequential (not enclosing the metadirective).
  ! The second parallel encloses the metadirective and should match.
  !$omp parallel do
  do i = 1, 10
  end do
  !$omp end parallel do
  !$omp parallel
  !$omp metadirective &
  !$omp & when(construct={parallel}: barrier) &
  !$omp & otherwise(nothing)
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPtest_construct_not_enclosing()
! CHECK-NOT:     omp.barrier
! CHECK:         return
subroutine test_construct_not_enclosing()
  ! No enclosing parallel — construct={parallel} should NOT match.
  !$omp metadirective &
  !$omp & when(construct={parallel}: barrier) &
  !$omp & otherwise(nothing)
end subroutine
