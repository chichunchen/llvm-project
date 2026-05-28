! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %s 2>&1 | FileCheck %s

subroutine target_map_iterator_component()
  type t
    integer :: a(10)
  end type
  type(t) :: x
  integer :: i

  ! CHECK: not yet implemented: target map iterator modifier for derived type components
  !$omp target map(iterator(i = 1:10), tofrom: x%a(i))
    x%a(1) = 1
  !$omp end target
end subroutine
