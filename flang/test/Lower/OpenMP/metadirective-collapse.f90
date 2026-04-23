! Test lowering of metadirective with collapse clause and inner metadirectives
! between collapsed loops.

! RUN: %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=52 %s -o - | FileCheck %s

! Multiple false conditions with default target teams loop collapse(2).
! Inner metadirective (resolving to nothing) sits between the two DO loops.
! The first two tests check precise host_eval forwarding to collapsed bounds;
! later tests only need to verify the selected collapsed construct shape.
! CHECK-LABEL: func.func @_QPtest_collapse_default_target_loop(
! CHECK:         omp.target_data
! CHECK:           omp.target host_eval(
! CHECK-SAME:        %{{[^[:space:],]+}} -> %[[DTL_LB1:[^,[:space:]]+]],
! CHECK-SAME:        %{{[^[:space:],]+}} -> %[[DTL_LB2:[^,[:space:]]+]],
! CHECK-SAME:        %{{[^[:space:],]+}} -> %[[DTL_UB1:[^,[:space:]]+]],
! CHECK-SAME:        %{{[^[:space:],]+}} -> %[[DTL_UB2:[^,[:space:]]+]],
! CHECK-SAME:        %{{[^[:space:],]+}} -> %[[DTL_STEP1:[^,[:space:]]+]],
! CHECK-SAME:        %{{[^[:space:],]+}} -> %[[DTL_STEP2:[^,[:space:]]+]]
! CHECK-SAME:        : {{.*}})
! CHECK:             omp.teams
! CHECK:               omp.parallel
! CHECK:                 omp.distribute
! CHECK:                   omp.wsloop
! CHECK-NOT:                 omp.simd
! CHECK:                       omp.loop_nest (%{{.*}}, %{{.*}}) : {{.*}} =
! CHECK-SAME:                    (%[[DTL_LB1]], %[[DTL_LB2]])
! CHECK-SAME:                    to (%[[DTL_UB1]], %[[DTL_UB2]]) inclusive
! CHECK-SAME:                    step (%[[DTL_STEP1]], %[[DTL_STEP2]]) collapse(2)
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
! CHECK:           omp.target host_eval(
! CHECK-SAME:        %{{[^[:space:],]+}} -> %[[TTD_LB1:[^,[:space:]]+]],
! CHECK-SAME:        %{{[^[:space:],]+}} -> %[[TTD_LB2:[^,[:space:]]+]],
! CHECK-SAME:        %{{[^[:space:],]+}} -> %[[TTD_UB1:[^,[:space:]]+]],
! CHECK-SAME:        %{{[^[:space:],]+}} -> %[[TTD_UB2:[^,[:space:]]+]],
! CHECK-SAME:        %{{[^[:space:],]+}} -> %[[TTD_STEP1:[^,[:space:]]+]],
! CHECK-SAME:        %{{[^[:space:],]+}} -> %[[TTD_STEP2:[^,[:space:]]+]]
! CHECK-SAME:        : {{.*}})
! CHECK:             omp.teams
! CHECK:               omp.parallel
! CHECK:                 omp.distribute
! CHECK:                   omp.wsloop
! CHECK:                     omp.simd
! CHECK:                       omp.loop_nest (%{{.*}}, %{{.*}}) : {{.*}} =
! CHECK-SAME:                    (%[[TTD_LB1]], %[[TTD_LB2]])
! CHECK-SAME:                    to (%[[TTD_UB1]], %[[TTD_UB2]]) inclusive
! CHECK-SAME:                    step (%[[TTD_STEP1]], %[[TTD_STEP2]]) collapse(2)
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
