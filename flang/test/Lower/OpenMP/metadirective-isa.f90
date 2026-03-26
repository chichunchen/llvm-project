! Test metadirective with device={isa(...)} trait selectors.
! Verifies that ISA traits are matched against the module's target features.

! RUN: %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=52 -target-feature +neon %s -o - | FileCheck --check-prefix=NEON %s
! RUN: %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=52 %s -o - | FileCheck --check-prefix=NO-NEON %s

! NEON-LABEL: func.func @_QPtest_isa_match()
! NEON:         omp.barrier
subroutine test_isa_match()
  ! When +neon is in target features, isa("neon") matches → barrier.
  !$omp metadirective &
  !$omp & when(device={isa("neon")}: barrier) &
  !$omp & otherwise(nothing)
end subroutine

! NO-NEON-LABEL: func.func @_QPtest_isa_match()
! NO-NEON-NOT:     omp.barrier
! NO-NEON:         return
