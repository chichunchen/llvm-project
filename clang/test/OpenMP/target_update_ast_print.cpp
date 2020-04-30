// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=50 -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=50 -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s

// RUN: %clang_cc1 -DOMP5 -verify -fopenmp -fopenmp-version=50 -ast-print %s | FileCheck %s --check-prefix=OMP5
// RUN: %clang_cc1 -DOMP5 -fopenmp -fopenmp-version=50 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s --check-prefix=OMP5

// RUN: %clang_cc1 -DOMP5 -verify -fopenmp-simd -fopenmp-version=50 -ast-print %s | FileCheck %s --check-prefix=OMP5
// RUN: %clang_cc1 -DOMP5 -fopenmp-simd -fopenmp-version=50 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -DOMP5 -fopenmp-simd -fopenmp-version=50 -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s --check-prefix=OMP5
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

void foo() {}

template <class T, class U>
T foo(T targ, U uarg) {
  static T a, *p;
  U b;
  int l;
#pragma omp target update to(([a][targ])p, a) if(l>5) device(l) nowait depend(inout:l)

#pragma omp target update from(b, ([a][targ])p) if(l<5) device(l-1) nowait depend(inout:l)

#ifdef OMP5
  U marr[10][10][10];
#pragma omp target update to(marr[2][0:2][0:2])

#pragma omp target update from(marr[2][0:2][0:2])

#pragma omp target update to(marr[:][0:2][0:2])

#pragma omp target update from(marr[:][0:2][0:2])

#pragma omp target update to(marr[:][:l][l:])

#pragma omp target update from(marr[:][:l][l:])

#pragma omp target update to(marr[:2][:1][:])

#pragma omp target update from(marr[:2][:1][:])

#pragma omp target update to(marr[:2][:][:1])

#pragma omp target update from(marr[:2][:][:1])

#pragma omp target update to(marr[:2][:][1:])

#pragma omp target update from(marr[:2][:][1:])

#pragma omp target update to(marr[:1][3:2][:2])

#pragma omp target update from(marr[:1][3:2][:2])

#pragma omp target update to(marr[:1][:2][0])

#pragma omp target update from(marr[:1][:2][0])

// OMP5: marr[10][10][10];
// OMP5-NEXT: #pragma omp target update to(marr[2][0:2][0:2])
// OMP5-NEXT: #pragma omp target update from(marr[2][0:2][0:2])
// OMP5-NEXT: #pragma omp target update to(marr[:][0:2][0:2])
// OMP5-NEXT: #pragma omp target update from(marr[:][0:2][0:2])
// OMP5-NEXT: #pragma omp target update to(marr[:][:l][l:])
// OMP5-NEXT: #pragma omp target update from(marr[:][:l][l:])
// OMP5-NEXT: #pragma omp target update to(marr[:2][:1][:])
// OMP5-NEXT: #pragma omp target update from(marr[:2][:1][:])
// OMP5-NEXT: #pragma omp target update to(marr[:2][:][:1])
// OMP5-NEXT: #pragma omp target update from(marr[:2][:][:1])
// OMP5-NEXT: #pragma omp target update to(marr[:2][:][1:])
// OMP5-NEXT: #pragma omp target update from(marr[:2][:][1:])
// OMP5-NEXT: #pragma omp target update to(marr[:1][3:2][:2])
// OMP5-NEXT: #pragma omp target update from(marr[:1][3:2][:2])
// OMP5-NEXT: #pragma omp target update to(marr[:1][:2][0])
// OMP5-NEXT: #pragma omp target update from(marr[:1][:2][0])
#endif

  return a + targ + (T)b;
}
// CHECK:      static T a, *p;
// CHECK-NEXT: U b;

int main(int argc, char **argv) {
  static int a;
  int n;
  float f;

// CHECK:      static int a;
// CHECK-NEXT: int n;
// CHECK-NEXT: float f;
#pragma omp target update to(a) if(f>0.0) device(n) nowait depend(in:n)
// CHECK-NEXT: #pragma omp target update to(a) if(f > 0.) device(n) nowait depend(in : n)
#pragma omp target update from(f) if(f<0.0) device(n+1) nowait depend(in:n)
// CHECK-NEXT: #pragma omp target update from(f) if(f < 0.) device(n + 1) nowait depend(in : n)

#ifdef OMP5
  float marr[10][10][10];
// OMP5: marr[10][10][10];
#pragma omp target update to(marr[2][0:2][0:2])
// OMP5-NEXT: #pragma omp target update to(marr[2][0:2][0:2])
#pragma omp target update from(marr[2][0:2][0:2])
// OMP5-NEXT: #pragma omp target update from(marr[2][0:2][0:2])
#pragma omp target update to(marr[:][0:2][0:2])
// OMP5-NEXT: #pragma omp target update to(marr[:][0:2][0:2])
#pragma omp target update from(marr[:][0:2][0:2])
// OMP5-NEXT: #pragma omp target update from(marr[:][0:2][0:2])
#pragma omp target update to(marr[:][:n][n:])
// OMP5: #pragma omp target update to(marr[:][:n][n:])
#pragma omp target update from(marr[:2][:1][:])
// OMP5-NEXT: #pragma omp target update from(marr[:2][:1][:])
#pragma omp target update to(marr[:2][:][:1])
// OMP5-NEXT: #pragma omp target update to(marr[:2][:][:1])
#pragma omp target update from(marr[:2][:][:1])
// OMP5-NEXT: #pragma omp target update from(marr[:2][:][:1])
#pragma omp target update to(marr[:2][:][1:])
// OMP5-NEXT: #pragma omp target update to(marr[:2][:][1:])
#pragma omp target update from(marr[:2][:][1:])
// OMP5-NEXT: #pragma omp target update from(marr[:2][:][1:])
#pragma omp target update to(marr[:1][3:2][:2])
// OMP5-NEXT: #pragma omp target update to(marr[:1][3:2][:2])
#pragma omp target update from(marr[:1][3:2][:2])
// OMP5-NEXT: #pragma omp target update from(marr[:1][3:2][:2])
#pragma omp target update to(marr[:1][:2][0])
// OMP5-NEXT: #pragma omp target update to(marr[:1][:2][0])
#pragma omp target update from(marr[:1][:2][0])
// OMP5-NEXT: #pragma omp target update from(marr[:1][:2][0])
#endif

  return foo(argc, f) + foo(argv[0][0], f) + a;
}

#endif
