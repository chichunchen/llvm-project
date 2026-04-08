! Test lowering of metadirective with loop-associated directive variants.
! Verifies that the sibling DO loop is correctly consumed by the
! loop-associated OpenMP construct.

! RUN: %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=52 %s -o - | FileCheck %s

! Static resolution: vendor matches -> parallel do with the loop.
! CHECK-LABEL: func @_QPtest_loop_static
! CHECK:         omp.parallel
! CHECK:           omp.wsloop
! CHECK:             omp.loop_nest
! CHECK-NOT:    fir.if
subroutine test_loop_static(a, n)
  real, intent(inout) :: a(:)
  integer, intent(in) :: n
  !$omp metadirective &
  !$omp   when(implementation={vendor(llvm)}: parallel do) &
  !$omp   otherwise(do)
  do i = 1, n
    a(i) = a(i) + 1.0
  end do
end subroutine
