! Offloading test for iterator modifier on map and motion clauses.

! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic

program map_motion_iterator
  implicit none
  integer, parameter :: n = 8
  type :: cell
    integer :: v
  end type cell
  !$omp declare mapper(cell_mapper : cell :: cc) map(tofrom: cc%v)
  integer :: a(n), b(n), c(n), d(n), e(n)
  integer :: r(n), t(n), u(n), v(n)
  integer :: i, dyn_n, dyn_step
  logical :: use_device
  type(cell) :: cells(n)

  do i = 1, n
    a(i) = i
  end do
  !$omp target enter data map(to: a)
  !$omp target
    do i = 2, n, 2
      a(i) = 100 + i
    end do
  !$omp end target
  do i = 2, n, 2
    a(i) = -100
  end do
  !$omp target update from(iterator(i = 2:n:2): a(i))
  !$omp target exit data map(delete: a)

  ! CHECK: update from: 1 102 3 104 5 106 7 108
  print *, "update from:", a

  do i = 1, n
    b(i) = i
  end do
  !$omp target enter data map(to: b)
  do i = 2, n, 2
    b(i) = 200 + i
  end do
  !$omp target update to(iterator(i = 2:n:2): b(i))
  !$omp target
    do i = 2, n, 2
      b(i) = b(i) + 10
    end do
  !$omp end target
  do i = 2, n, 2
    b(i) = -100
  end do
  !$omp target update from(iterator(i = 2:n:2): b(i))
  !$omp target exit data map(delete: b)

  ! CHECK: update tofrom: 1 212 3 214 5 216 7 218
  print *, "update tofrom:", b

  do i = 1, n
    c(i) = i
  end do
  !$omp target enter data map(to: c)
  !$omp target
    c(1) = 11
    c(3) = 33
    c(5) = 55
    c(7) = 77
  !$omp end target
  do i = 1, n, 2
    c(i) = -100
  end do
  !$omp target exit data map(iterator(i = 1:n:2), from: c(i))
  !$omp target exit data map(delete: c)

  ! CHECK: exit data: 11 2 33 4 55 6 77 8
  print *, "exit data:", c

  do i = 1, n
    d(i) = i
  end do
  !$omp target data map(iterator(i = 1:n:2), tofrom: d(i))
    !$omp target map(present, alloc: d(1))
      d(1) = 21
    !$omp end target
    !$omp target map(present, alloc: d(3))
      d(3) = 43
    !$omp end target
    !$omp target map(present, alloc: d(5))
      d(5) = 65
    !$omp end target
    !$omp target map(present, alloc: d(7))
      d(7) = 87
    !$omp end target
  !$omp end target data

  ! CHECK: target data: 21 2 43 4 65 6 87 8
  print *, "target data:", d

  dyn_n = n
  dyn_step = 2
  use_device = .true.
  do i = 1, n
    e(i) = i
  end do
  !$omp target data if(use_device) &
  !$omp& map(iterator(i = 1:dyn_n:dyn_step), tofrom: e(i))
    !$omp target map(present, alloc: e(1))
      e(1) = 31
    !$omp end target
    !$omp target map(present, alloc: e(3))
      e(3) = 53
    !$omp end target
    !$omp target map(present, alloc: e(5))
      e(5) = 75
    !$omp end target
    !$omp target map(present, alloc: e(7))
      e(7) = 97
    !$omp end target
  !$omp end target data

  ! CHECK: target data if: 31 2 53 4 75 6 97 8
  print *, "target data if:", e

  ! A user-defined mapper combined with the iterator modifier: the mapper is
  ! resolved for each iterator-expanded entry of an array of a derived type.
  ! The mapper maps the whole `cell` through its only member `v`, so the
  ! per-element `map(present, alloc: cells(k))` below resolves to the same
  ! device allocation the mapper registered (cell has a single member at
  ! offset 0, so the object and the member cover the same storage).
  do i = 1, n
    cells(i)%v = i
  end do
  !$omp target data map(mapper(cell_mapper), iterator(i = 1:n:2), tofrom: cells(i))
    !$omp target map(present, alloc: cells(1))
      cells(1)%v = 21
    !$omp end target
    !$omp target map(present, alloc: cells(3))
      cells(3)%v = 43
    !$omp end target
    !$omp target map(present, alloc: cells(5))
      cells(5)%v = 65
    !$omp end target
    !$omp target map(present, alloc: cells(7))
      cells(7)%v = 87
    !$omp end target
  !$omp end target data

  ! CHECK: declare mapper iter: 21 2 43 4 65 6 87 8
  print *, "declare mapper iter:", cells%v

  ! Stress: a whole-array static map and an iterator-expanded element map in the
  ! same target enter data, updated back with a static and an iterator motion.
  do i = 1, n
    r(i) = i
    t(i) = i
  end do
  !$omp target enter data map(to: r) map(iterator(i = 1:n:2), to: t(i))
  !$omp target map(present, alloc: r)
    do i = 1, n
      r(i) = 800 + i
    end do
  !$omp end target
  !$omp target map(present, alloc: t(1))
    t(1) = 91
  !$omp end target
  !$omp target map(present, alloc: t(3))
    t(3) = 93
  !$omp end target
  !$omp target map(present, alloc: t(5))
    t(5) = 95
  !$omp end target
  !$omp target map(present, alloc: t(7))
    t(7) = 97
  !$omp end target
  do i = 1, n
    r(i) = -100
  end do
  do i = 1, n, 2
    t(i) = -100
  end do
  !$omp target update from(r) from(iterator(i = 1:n:2): t(i))
  !$omp target exit data map(delete: r) map(delete: t)

  ! CHECK: mixed static+iter r: 801 802 803 804 805 806 807 808
  print *, "mixed static+iter r:", r
  ! CHECK: mixed static+iter t: 91 2 93 4 95 6 97 8
  print *, "mixed static+iter t:", t

  ! Stress: two motion iterators with disjoint ranges in one update directive
  ! (odd elements of u, even elements of v).
  do i = 1, n
    u(i) = i
    v(i) = i
  end do
  !$omp target enter data map(to: u) map(to: v)
  !$omp target map(present, alloc: u) map(present, alloc: v)
    do i = 1, n, 2
      u(i) = 600 + i
    end do
    do i = 2, n, 2
      v(i) = 700 + i
    end do
  !$omp end target
  do i = 1, n, 2
    u(i) = -100
  end do
  do i = 2, n, 2
    v(i) = -100
  end do
  !$omp target update from(iterator(i = 1:n:2): u(i)) &
  !$omp&                from(iterator(i = 2:n:2): v(i))
  !$omp target exit data map(delete: u) map(delete: v)

  ! CHECK: multi iter u: 601 2 603 4 605 6 607 8
  print *, "multi iter u:", u
  ! CHECK: multi iter v: 1 702 3 704 5 706 7 708
  print *, "multi iter v:", v

end program map_motion_iterator
