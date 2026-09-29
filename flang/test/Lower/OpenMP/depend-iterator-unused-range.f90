! RUN: %flang_fc1 -emit-llvm -O1 -fopenmp -fopenmp-version=52 -o - %s \
! RUN:   | FileCheck %s

! The unused j range must not affect the number of dependences, including
! when m < 3 makes that range empty.
subroutine depend_unused_iterator(a, m)
  integer :: a(2), m

  !$omp task depend(iterator(i = 1:2, j = 3:m), in: a(i))
  !$omp end task
end subroutine

! CHECK-LABEL: define {{.*}} @depend_unused_iterator_(
! CHECK: call i32 @__kmpc_omp_task_with_deps(
! CHECK-SAME: ptr {{[^,]+}}, i32 {{[^,]+}}, ptr {{[^,]+}}, i32 2,

! Affinity uses the same per-locator selection and must also ignore empty j.
subroutine affinity_unused_iterator(a, m)
  integer :: a(2), m

  !$omp task affinity(iterator(i = 1:2, j = 3:m): a(i))
  !$omp end task
end subroutine

! CHECK-LABEL: define {{.*}} @affinity_unused_iterator_(
! CHECK: call i32 @__kmpc_omp_reg_task_with_affinity(
! CHECK-SAME: ptr {{[^,]+}}, i32 {{[^,]+}}, ptr {{[^,]+}}, i32 2,
