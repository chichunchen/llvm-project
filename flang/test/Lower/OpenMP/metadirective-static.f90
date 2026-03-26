! Test lowering of OpenMP metadirective with static context selectors.
! Verifies that metadirective is resolved at compile time and only
! the selected directive variant's MLIR ops are emitted.

! RUN: %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=52 %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPtest_vendor_llvm()
! CHECK:         omp.barrier
! CHECK:         return
subroutine test_vendor_llvm()
  ! Vendor(llvm) should match — emit barrier.
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: barrier) &
  !$omp & otherwise(nothing)
end subroutine

! CHECK-LABEL: func.func @_QPtest_vendor_other()
! CHECK-NOT:     omp.barrier
! CHECK:         return
subroutine test_vendor_other()
  ! Vendor(amd) should not match — fall through to otherwise(nothing).
  !$omp metadirective &
  !$omp & when(implementation={vendor("amd")}: barrier) &
  !$omp & otherwise(nothing)
end subroutine

! CHECK-LABEL: func.func @_QPtest_device_kind_host()
! CHECK:         omp.barrier
! CHECK:         return
subroutine test_device_kind_host()
  ! device={kind(host)} should match in host compilation.
  !$omp metadirective &
  !$omp & when(device={kind(host)}: barrier) &
  !$omp & otherwise(nothing)
end subroutine

! CHECK-LABEL: func.func @_QPtest_nothing_variant()
! CHECK-NOT:     omp.barrier
! CHECK:         return
subroutine test_nothing_variant()
  ! when(... : nothing) should emit no OpenMP ops.
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: nothing) &
  !$omp & otherwise(barrier)
end subroutine

! CHECK-LABEL: func.func @_QPtest_otherwise_directive()
! CHECK:         omp.barrier
! CHECK:         return
subroutine test_otherwise_directive()
  ! No when clause matches — fall through to otherwise(barrier).
  !$omp metadirective &
  !$omp & when(implementation={vendor("amd")}: nothing) &
  !$omp & otherwise(barrier)
end subroutine

! CHECK-LABEL: func.func @_QPtest_no_otherwise()
! CHECK-NOT:     omp.barrier
! CHECK:         return
subroutine test_no_otherwise()
  ! No clause matches and no otherwise — implicit nothing.
  !$omp metadirective &
  !$omp & when(implementation={vendor("amd")}: barrier)
end subroutine

! CHECK-LABEL: func.func @_QPtest_condition_true()
! CHECK:         omp.barrier
! CHECK-NOT:     fir.if
! CHECK:         return
subroutine test_condition_true()
  ! Constant .true. condition is resolved statically.
  !$omp metadirective &
  !$omp & when(user={condition(.true.)}: barrier) &
  !$omp & otherwise(nothing)
end subroutine

! CHECK-LABEL: func.func @_QPtest_condition_false()
! CHECK-NOT:     omp.barrier
! CHECK-NOT:     fir.if
! CHECK:         return
subroutine test_condition_false()
  ! Constant .false. condition is resolved statically.
  !$omp metadirective &
  !$omp & when(user={condition(.false.)}: barrier) &
  !$omp & otherwise(nothing)
end subroutine
