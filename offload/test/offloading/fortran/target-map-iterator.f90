! Offloading test for iterator modifier on target map clauses.

! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic

program target_map_iterator
  implicit none
  integer, parameter :: n = 8
  integer :: a(n), b(n), c(n), d(n), e(n), f(n)
  integer :: i, dyn_n, dyn_step

  do i = 1, n
    a(i) = i
    b(i) = i
    c(i) = i
    d(i) = i
    e(i) = i
    f(i) = i
  end do

  ! Touch the first mapped element so the test exercises target map iterator
  ! lowering without depending on dynamic per-entry lookup inside the kernel.
  !$omp target map(iterator(i = 1:n), tofrom: a(i))
    a(1) = 101
  !$omp end target

  ! CHECK: static: 101 2 3 4 5 6 7 8
  print *, "static:", a

  dyn_n = n
  dyn_step = 2
  !$omp target map(iterator(i = 1:dyn_n:dyn_step), tofrom: b(i))
    b(1) = 201
  !$omp end target

  ! CHECK: dynamic: 201 2 3 4 5 6 7 8
  print *, "dynamic:", b

  !$omp target map(iterator(i = 1:n), tofrom: c(i), d(i))
    c(1) = 301
    d(1) = 302
  !$omp end target

  ! CHECK: multi c: 301 2 3 4 5 6 7 8
  print *, "multi c:", c
  ! CHECK: multi d: 302 2 3 4 5 6 7 8
  print *, "multi d:", d

  !$omp target map(iterator(i = 2:n), tofrom: e(1), e(i))
    e(1) = 401
  !$omp end target

  ! CHECK: mixed: 401 2 3 4 5 6 7 8
  print *, "mixed:", e

  !$omp target map(iterator(i = 1:n - 1), tofrom: f(i), f(i + 1))
    f(1) = 501
  !$omp end target

  ! CHECK: shared capture: 501 2 3 4 5 6 7 8
  print *, "shared capture:", f
end program target_map_iterator
