! Test lowering of metadirective with collapse clause and inner metadirectives
! between collapsed loops.

! RUN: %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=52 %s -o - | FileCheck %s

! Multiple false conditions with default target teams loop collapse(2).
! Inner metadirective (resolving to nothing) sits between the two DO loops.
! CHECK-LABEL: func.func @_QPtest_collapse_default_target_loop(
! CHECK:         omp.target_data
! CHECK:           omp.target
! CHECK:             omp.teams
! CHECK:               omp.parallel
! CHECK:                 omp.distribute
! CHECK:                   omp.wsloop
! CHECK-NOT:                 omp.simd
! CHECK:                       omp.loop_nest {{.*}} collapse(2)
subroutine test_collapse_default_target_loop(n, a, b, c)
  real(8) :: a(n,n), b(n,n), c(n,n)
  !$omp target data map(tofrom: a, b, c)
  !$omp metadirective &
  !$omp & when(user={condition(.false.)}: parallel do) &
  !$omp & when(user={condition(.false.)}: target teams distribute parallel do simd collapse(2)) &
  !$omp & when(user={condition(.false.)}: target teams distribute parallel do) &
  !$omp & default(target teams loop collapse(2))
  do j = 1, n
  !$omp metadirective when(user={condition(.false.)}: simd)
    do i = 1, n
      a(i,j) = b(i,j) + c(i,j)
    enddo
  enddo
  !$omp end target data
end subroutine

! CHECK-LABEL: func.func @_QPtest_collapse_true_target_distrib(
! CHECK:         omp.target_data
! CHECK:           omp.target
! CHECK:             omp.teams
! CHECK:               omp.parallel
! CHECK:                 omp.distribute
! CHECK:                   omp.wsloop
! CHECK:                     omp.simd
! CHECK:                       omp.loop_nest {{.*}} collapse(2)
subroutine test_collapse_true_target_distrib(n, a, b, c)
  real(8) :: a(n,n), b(n,n), c(n,n)
  !$omp target data map(tofrom: a, b, c)
  !$omp metadirective &
  !$omp & when(user={condition(.true.)}: target teams distribute parallel do simd collapse(2)) &
  !$omp & default(parallel do)
  do j = 1, n
    do i = 1, n
      a(i,j) = b(i,j) + c(i,j)
    enddo
  enddo
  !$omp end target data
end subroutine

! CHECK-LABEL: func.func @_QPtest_collapse_default_only(
! CHECK:         omp.target_data
! CHECK:           omp.target
! CHECK:             omp.teams
! CHECK:               omp.parallel
! CHECK:                 omp.distribute
! CHECK:                   omp.wsloop
! CHECK:                     omp.loop_nest {{.*}} collapse(2)
subroutine test_collapse_default_only(n, a, b, c)
  real(8) :: a(n,n), b(n,n), c(n,n)
  !$omp target data map(tofrom: a, b, c)
  !$omp metadirective &
  !$omp & default(target teams loop collapse(2))
  do j = 1, n
  !$omp metadirective when(user={condition(.false.)}: simd)
    do i = 1, n
      a(i,j) = b(i,j) + c(i,j)
    enddo
  enddo
  !$omp end target data
end subroutine

! CHECK-LABEL: func.func @_QPtest_collapse_true_target_parallel(
! CHECK:         omp.target_data
! CHECK:           omp.target
! CHECK:             omp.teams
! CHECK:               omp.parallel
! CHECK:                 omp.distribute
! CHECK:                   omp.wsloop
! CHECK:                     omp.loop_nest {{.*}} collapse(2)
subroutine test_collapse_true_target_parallel(n, a, b, c)
  real(8) :: a(n,n), b(n,n), c(n,n)
  !$omp target data map(tofrom: a, b, c)
  !$omp metadirective &
  !$omp & when(user={condition(.true.)}: target teams distribute parallel do collapse(2)) &
  !$omp & default(parallel do)
  do j = 1, n
    do i = 1, n
      a(i,j) = b(i,j) + c(i,j)
    enddo
  enddo
  !$omp end target data
end subroutine

! CHECK-LABEL: func.func @_QPtest_collapse_multi_condition(
! CHECK:         omp.target_data
! CHECK:           omp.target
! CHECK:             omp.teams
! CHECK:               omp.parallel
! CHECK:                 omp.distribute
! CHECK:                   omp.wsloop
! CHECK:                     omp.simd
! CHECK:                       omp.loop_nest {{.*}} collapse(2)
subroutine test_collapse_multi_condition(n, a, b, c)
  real(8) :: a(n,n), b(n,n), c(n,n)
  !$omp target data map(tofrom: a, b, c)
  !$omp metadirective &
  !$omp & when(user={condition(.false.)}: parallel do) &
  !$omp & when(user={condition(.true.)}: target teams distribute parallel do simd collapse(2)) &
  !$omp & default(target teams loop collapse(2))
  do j = 1, n
  !$omp metadirective when(user={condition(.false.)}: simd)
    do i = 1, n
      a(i,j) = b(i,j) + c(i,j)
    enddo
  enddo
  !$omp end target data
end subroutine
