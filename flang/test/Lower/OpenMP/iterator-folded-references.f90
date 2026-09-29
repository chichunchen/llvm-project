! RUN: bbc -emit-hlfir -fopenmp -fopenmp-version=52 -o - %s | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %s | FileCheck %s
! RUN: %flang_fc1 -emit-llvm -O1 -fopenmp -fopenmp-version=52 -o - %s | FileCheck %s --check-prefix=LLVM

! Even though folding removes 0*i, an empty i range must prevent evaluation
! of j/d. In particular, m=0, d=0 must not evaluate the division.
subroutine depend_folded_reference(a, m, d)
  integer :: a(2), m, d
  !$omp task depend(iterator(i=1:m, j=1:2), in: a(j/d + 0*i))
  !$omp end task
end
! LLVM-LABEL: define {{.*}} @depend_folded_reference_(
! LLVM: [[M:%.*]] = load i32, ptr
! LLVM: [[NONNEG:%.*]] = {{.*}}call i32 @llvm.smax.i32(i32 [[M]], i32 0)
! LLVM: [[COUNT:%.*]] = shl nuw i32 [[NONNEG]], 1
! LLVM: [[EMPTY:%.*]] = icmp slt i32 [[M]], 1
! LLVM-NEXT: br i1 [[EMPTY]], label %[[CONT:[^,]+]], label %[[BODY:.*]]
! LLVM: [[BODY]]:
! LLVM: sdiv i32
! LLVM: [[CONT]]:
! LLVM: call i32 @__kmpc_omp_task_with_deps(ptr {{[^,]+}}, i32 {{[^,]+}}, ptr {{[^,]+}}, i32 [[COUNT]],

! CHECK-LABEL: func.func @_QPdepend_folded_reference(
! CHECK: omp.iterator(%{{[^:]+}}: index, %{{[^:]+}}: index) =
! CHECK: arith.divsi
! CHECK: omp.yield
! CHECK: omp.task depend(

subroutine affinity_folded_reference(a, m, d)
  integer :: a(2), m, d
  !$omp task affinity(iterator(i=1:m, j=1:2): a(j/d + 0*i))
  !$omp end task
end
! LLVM-LABEL: define {{.*}} @affinity_folded_reference_(
! LLVM: [[M:%.*]] = load i32, ptr
! LLVM: [[NONNEG:%.*]] = {{.*}}call i32 @llvm.smax.i32(i32 [[M]], i32 0)
! LLVM: [[COUNT:%.*]] = shl nuw i32 [[NONNEG]], 1
! LLVM: [[EMPTY:%.*]] = icmp slt i32 [[M]], 1
! LLVM-NEXT: br i1 [[EMPTY]], label %[[CONT:[^,]+]], label %[[BODY:.*]]
! LLVM: [[BODY]]:
! LLVM: sdiv i32
! LLVM: [[CONT]]:
! LLVM: call i32 @__kmpc_omp_reg_task_with_affinity(ptr {{[^,]+}}, i32 {{[^,]+}}, ptr {{[^,]+}}, i32 [[COUNT]],

! CHECK-LABEL: func.func @_QPaffinity_folded_reference(
! CHECK: omp.iterator(%{{[^:]+}}: index, %{{[^:]+}}: index) =
! CHECK: arith.divsi
! CHECK: omp.affinity_entry
! CHECK: omp.yield
! CHECK: omp.task affinity(

! These locators have equal folded designators but distinct source references.
! The last one references no iterator; the middle one references both even
! though all of its iterator references disappear during folding.
subroutine depend_equal_designators(a, m)
  integer :: a(2), m
  !$omp task depend(iterator(i=1:m, j=1:2), in: a(1+0*j), a(1+0*j+0*i), a(1))
  !$omp end task
end
! CHECK-LABEL: func.func @_QPdepend_equal_designators(
! CHECK: omp.iterator(%{{[^:]+}}: index) =
! CHECK: omp.yield
! CHECK: omp.iterator(%{{[^:]+}}: index, %{{[^:]+}}: index) =
! CHECK: omp.yield
! CHECK-NOT: omp.iterator
! CHECK: omp.task depend(

subroutine affinity_equal_designators(a, m)
  integer :: a(2), m
  !$omp task affinity(iterator(i=1:m, j=1:2): a(1+0*j), a(1+0*j+0*i), a(1))
  !$omp end task
end
! CHECK-LABEL: func.func @_QPaffinity_equal_designators(
! CHECK: omp.iterator(%{{[^:]+}}: index) =
! CHECK: omp.yield
! CHECK: omp.iterator(%{{[^:]+}}: index, %{{[^:]+}}: index) =
! CHECK: omp.yield
! CHECK-NOT: omp.iterator
! CHECK: omp.task affinity(

! Identically spelled iterator names in different clauses have distinct scopes.
subroutine depend_scoped_references(a, m)
  integer :: a(2), m
  !$omp task depend(iterator(i=1:m), in: a(1+0*i)) &
  !$omp& depend(iterator(i=1:2), out: a(i))
  !$omp end task
end
! CHECK-LABEL: func.func @_QPdepend_scoped_references(
! CHECK: omp.iterator(%{{[^:]+}}: index) =
! CHECK: omp.yield
! CHECK: omp.iterator(%{{[^:]+}}: index) =
! CHECK: omp.yield
! CHECK: omp.task depend(

! Per-locator source metadata must also survive target clause decomposition.
subroutine target_folded_reference(a, m)
  integer :: a(2), m
  !$omp target parallel do map(tofrom:a) depend(iterator(i=1:m), in: a(1+0*i))
  do k=1,2
    a(k)=k
  end do
  !$omp end target parallel do
  !$omp target enter data map(to:a) depend(iterator(i=1:m), in: a(1+0*i))
  !$omp target update to(a) depend(iterator(i=1:m), in: a(1+0*i))
  !$omp target exit data map(from:a) depend(iterator(i=1:m), in: a(1+0*i))
end
! CHECK-LABEL: func.func @_QPtarget_folded_reference(
! CHECK: omp.iterator(%{{[^:]+}}: index) =
! CHECK: omp.yield
! CHECK: omp.target
! CHECK: omp.iterator(%{{[^:]+}}: index) =
! CHECK: omp.yield
! CHECK: omp.target_enter_data
! CHECK: omp.iterator(%{{[^:]+}}: index) =
! CHECK: omp.yield
! CHECK: omp.target_update
! CHECK: omp.iterator(%{{[^:]+}}: index) =
! CHECK: omp.yield
! CHECK: omp.target_exit_data
