! Test that DEFAULT clause on METADIRECTIVE produces a deprecation warning
! for OpenMP >= 5.2, and no warning for OpenMP < 5.2.

! RUN: %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=52 %s -o /dev/null 2>&1 | FileCheck --check-prefix=WARN %s
! RUN: %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=50 %s -o /dev/null 2>&1 | FileCheck --check-prefix=NO-WARN %s

! WARN: warning:{{.*}}DEFAULT clause on METADIRECTIVE is deprecated in OpenMP 5.2; use OTHERWISE instead
! NO-WARN-NOT: deprecated

subroutine test()
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: barrier) &
  !$omp & default(nothing)
end subroutine
