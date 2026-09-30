! RUN: %flang_fc1 -emit-llvm -O1 -fopenmp -fopenmp-version=52 -o - %s | FileCheck %s

! Empty ranges must produce zero entries and skip locator evaluation. Test
! dynamic bounds because the omp.iterator verifier rejects constant empty
! ranges. Bounds equal to the initial value must still produce one entry.
subroutine depend_empty_positive(a, m, d)
  integer :: a(8), m, d
  !$omp task depend(iterator(i=3:m), in: a(i/d))
  !$omp end task
end
! CHECK-LABEL: define {{.*}} @depend_empty_positive_(
! CHECK: [[M:%.*]] = load i32, ptr
! CHECK: [[WIDE:%.*]] = sext i32 [[M]] to i64
! CHECK: [[EMPTY:%.*]] = icmp slt i32 [[M]], 3
! CHECK: [[NONEMPTY:%.*]] = add nsw i64 [[WIDE]], -2
! CHECK: [[COUNT:%.*]] = select i1 [[EMPTY]], i64 0, i64 [[NONEMPTY]]
! CHECK: [[SIZE:%.*]] = mul {{.*}}i64 [[COUNT]], 24
! CHECK: [[LIST:%.*]] = {{.*}}call ptr @malloc(i64 [[SIZE]])
! CHECK: [[ZERO:%.*]] = icmp eq i64 [[COUNT]], 0
! CHECK-NEXT: br i1 [[ZERO]], label %[[CONT:[^,]+]], label %[[BODY:.*]]
! CHECK: [[BODY]]:
! CHECK: sdiv i32
! CHECK: [[CONT]]:
! CHECK: [[RUNTIME_COUNT:%.*]] = trunc i64 [[COUNT]] to i32
! CHECK: call i32 @__kmpc_omp_task_with_deps(ptr {{[^,]+}}, i32 {{[^,]+}}, ptr {{[^,]+}}, i32 [[RUNTIME_COUNT]], ptr {{[^,)]*}}[[LIST]]

subroutine affinity_empty_positive(a, m, d)
  integer :: a(8), m, d
  !$omp task affinity(iterator(i=3:m): a(i/d))
  !$omp end task
end
! CHECK-LABEL: define {{.*}} @affinity_empty_positive_(
! CHECK: [[M:%.*]] = load i32, ptr
! CHECK: [[WIDE:%.*]] = sext i32 [[M]] to i64
! CHECK: [[EMPTY:%.*]] = icmp slt i32 [[M]], 3
! CHECK: [[NONEMPTY:%.*]] = add nsw i64 [[WIDE]], -2
! CHECK: [[COUNT:%.*]] = select i1 [[EMPTY]], i64 0, i64 [[NONEMPTY]]
! CHECK: [[LIST:%.*]] = alloca { i64, i64, i32 }, i64 [[COUNT]]
! CHECK: [[ZERO:%.*]] = icmp eq i64 [[COUNT]], 0
! CHECK-NEXT: br i1 [[ZERO]], label %[[CONT:[^,]+]], label %[[BODY:.*]]
! CHECK: [[BODY]]:
! CHECK: sdiv i32
! CHECK: [[CONT]]:
! CHECK: [[RUNTIME_COUNT:%.*]] = trunc i64 [[COUNT]] to i32
! CHECK: call i32 @__kmpc_omp_reg_task_with_affinity(ptr {{[^,]+}}, i32 {{[^,]+}}, ptr {{[^,]+}}, i32 [[RUNTIME_COUNT]], ptr {{[^,)]*}}[[LIST]]

subroutine depend_empty_negative(a, m, d)
  integer :: a(8), m, d
  !$omp task depend(iterator(i=3:m:-2), in: a(i/d))
  !$omp end task
end
! CHECK-LABEL: define {{.*}} @depend_empty_negative_(
! CHECK: [[M:%.*]] = load i32, ptr
! CHECK: [[WIDE:%.*]] = sext i32 [[M]] to i65
! CHECK: [[SPAN:%.*]] = sub nsw i65 3, [[WIDE]]
! CHECK: [[EMPTY:%.*]] = icmp sgt i32 [[M]], 3
! CHECK: [[HALF:%.*]] = lshr i65 [[SPAN]], 1
! CHECK: [[HALF64:%.*]] = trunc {{.*}}i65 [[HALF]] to i64
! CHECK: [[NONEMPTY:%.*]] = add i64 [[HALF64]], 1
! CHECK: [[COUNT:%.*]] = select i1 [[EMPTY]], i64 0, i64 [[NONEMPTY]]
! CHECK: [[SIZE:%.*]] = mul {{.*}}i64 [[COUNT]], 24
! CHECK: [[LIST:%.*]] = {{.*}}call ptr @malloc(i64 [[SIZE]])
! CHECK: [[ZERO:%.*]] = icmp eq i64 [[COUNT]], 0
! CHECK-NEXT: br i1 [[ZERO]], label %[[CONT:[^,]+]], label %[[BODY:.*]]
! CHECK: [[BODY]]:
! CHECK: sdiv i32
! CHECK: [[CONT]]:
! CHECK: [[RUNTIME_COUNT:%.*]] = trunc i64 [[COUNT]] to i32
! CHECK: call i32 @__kmpc_omp_task_with_deps(ptr {{[^,]+}}, i32 {{[^,]+}}, ptr {{[^,]+}}, i32 [[RUNTIME_COUNT]], ptr {{[^,)]*}}[[LIST]]

subroutine affinity_empty_negative(a, m, d)
  integer :: a(8), m, d
  !$omp task affinity(iterator(i=3:m:-2): a(i/d))
  !$omp end task
end
! CHECK-LABEL: define {{.*}} @affinity_empty_negative_(
! CHECK: [[M:%.*]] = load i32, ptr
! CHECK: [[WIDE:%.*]] = sext i32 [[M]] to i65
! CHECK: [[SPAN:%.*]] = sub nsw i65 3, [[WIDE]]
! CHECK: [[EMPTY:%.*]] = icmp sgt i32 [[M]], 3
! CHECK: [[HALF:%.*]] = lshr i65 [[SPAN]], 1
! CHECK: [[HALF64:%.*]] = trunc {{.*}}i65 [[HALF]] to i64
! CHECK: [[NONEMPTY:%.*]] = add i64 [[HALF64]], 1
! CHECK: [[COUNT:%.*]] = select i1 [[EMPTY]], i64 0, i64 [[NONEMPTY]]
! CHECK: [[LIST:%.*]] = alloca { i64, i64, i32 }, i64 [[COUNT]]
! CHECK: [[ZERO:%.*]] = icmp eq i64 [[COUNT]], 0
! CHECK-NEXT: br i1 [[ZERO]], label %[[CONT:[^,]+]], label %[[BODY:.*]]
! CHECK: [[BODY]]:
! CHECK: sdiv i32
! CHECK: [[CONT]]:
! CHECK: [[RUNTIME_COUNT:%.*]] = trunc i64 [[COUNT]] to i32
! CHECK: call i32 @__kmpc_omp_reg_task_with_affinity(ptr {{[^,]+}}, i32 {{[^,]+}}, ptr {{[^,]+}}, i32 [[RUNTIME_COUNT]], ptr {{[^,)]*}}[[LIST]]

! A zero count in either retained dimension must suppress the entire locator.
subroutine depend_empty_product(a, m, n, d)
  integer :: a(8,8), m, n, d
  !$omp task depend(iterator(i=3:m, j=1:n), in: a(i/d,j))
  !$omp end task
end
! CHECK-LABEL: define {{.*}} @depend_empty_product_(
! CHECK: [[M:%.*]] = load i32, ptr
! CHECK: [[N:%.*]] = load i32, ptr
! CHECK: [[WIDE:%.*]] = sext i32 [[M]] to i64
! CHECK: [[EMPTY:%.*]] = icmp slt i32 [[M]], 3
! CHECK: [[NONEMPTY:%.*]] = add nsw i64 [[WIDE]], -2
! CHECK: [[COUNT:%.*]] = select i1 [[EMPTY]], i64 0, i64 [[NONEMPTY]]
! CHECK: [[NCOUNT:%.*]] = {{.*}}call i32 @llvm.smax.i32(i32 [[N]], i32 0)
! CHECK: [[N64:%.*]] = zext {{.*}}i32 [[NCOUNT]] to i64
! CHECK: [[TOTAL:%.*]] = mul {{.*}}i64 [[COUNT]], [[N64]]
! CHECK: [[SIZE:%.*]] = mul {{.*}}i64 [[TOTAL]], 24
! CHECK: [[LIST:%.*]] = {{.*}}call ptr @malloc(i64 [[SIZE]])
! CHECK: [[ZERO:%.*]] = icmp eq i64 [[TOTAL]], 0
! CHECK-NEXT: br i1 [[ZERO]], label %[[CONT:[^,]+]], label %[[BODY:.*]]
! CHECK: [[BODY]]:
! CHECK: sdiv i32
! CHECK: [[CONT]]:
! CHECK: [[RUNTIME_COUNT:%.*]] = trunc i64 [[TOTAL]] to i32
! CHECK: call i32 @__kmpc_omp_task_with_deps(ptr {{[^,]+}}, i32 {{[^,]+}}, ptr {{[^,]+}}, i32 [[RUNTIME_COUNT]], ptr {{[^,)]*}}[[LIST]]

subroutine affinity_empty_product(a, m, n, d)
  integer :: a(8,8), m, n, d
  !$omp task affinity(iterator(i=3:m, j=1:n): a(i/d,j))
  !$omp end task
end
! CHECK-LABEL: define {{.*}} @affinity_empty_product_(
! CHECK: [[M:%.*]] = load i32, ptr
! CHECK: [[N:%.*]] = load i32, ptr
! CHECK: [[WIDE:%.*]] = sext i32 [[M]] to i64
! CHECK: [[EMPTY:%.*]] = icmp slt i32 [[M]], 3
! CHECK: [[NONEMPTY:%.*]] = add nsw i64 [[WIDE]], -2
! CHECK: [[COUNT:%.*]] = select i1 [[EMPTY]], i64 0, i64 [[NONEMPTY]]
! CHECK: [[NCOUNT:%.*]] = {{.*}}call i32 @llvm.smax.i32(i32 [[N]], i32 0)
! CHECK: [[N64:%.*]] = zext {{.*}}i32 [[NCOUNT]] to i64
! CHECK: [[TOTAL:%.*]] = mul {{.*}}i64 [[COUNT]], [[N64]]
! CHECK: [[LIST:%.*]] = alloca { i64, i64, i32 }, i64 [[TOTAL]]
! CHECK: [[ZERO:%.*]] = icmp eq i64 [[TOTAL]], 0
! CHECK-NEXT: br i1 [[ZERO]], label %[[CONT:[^,]+]], label %[[BODY:.*]]
! CHECK: [[BODY]]:
! CHECK: sdiv i32
! CHECK: [[CONT]]:
! CHECK: [[RUNTIME_COUNT:%.*]] = trunc i64 [[TOTAL]] to i32
! CHECK: call i32 @__kmpc_omp_reg_task_with_affinity(ptr {{[^,]+}}, i32 {{[^,]+}}, ptr {{[^,]+}}, i32 [[RUNTIME_COUNT]], ptr {{[^,)]*}}[[LIST]]
