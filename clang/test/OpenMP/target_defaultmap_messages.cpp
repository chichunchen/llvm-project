// RUN: %clang_cc1 -verify -fopenmp %s -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp-simd %s -Wuninitialized

struct Bar {
  int a;
};

class Baz {
  public:
    Bar bar;
    int *p;
};

int *g;
int arr[50];
Bar bar;
Baz baz;
Baz* bazPtr = &baz;

void foo() {
}

template <class T, typename S, int N, int ST>
T tmain(T argc, S **argv) {
  #pragma omp target defaultmap // expected-error {{expected '(' after 'defaultmap'}}
  foo();
  #pragma omp target defaultmap ( // expected-error {{expected ['alloc', 'from', 'to', 'tofrom', 'firstprivate', 'none', 'default'] in OpenMP clause 'defaultmap'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target defaultmap () // expected-error {{expected ['alloc', 'from', 'to', 'tofrom', 'firstprivate', 'none', 'default'] in OpenMP clause 'defaultmap'}}
  foo();
  #pragma omp target defaultmap (tofrom // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-warning {{missing ':' after defaultmap modifier - ignoring}} expected-error {{expected ['scalar', 'aggregate', 'pointer'] in OpenMP clause 'defaultmap'}}
  foo();
  #pragma omp target defaultmap (tofrom: // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expected ['scalar', 'aggregate', 'pointer'] in OpenMP clause 'defaultmap'}}
  foo();
  #pragma omp target defaultmap (tofrom) // expected-warning {{missing ':' after defaultmap modifier - ignoring}} expected-error {{expected ['scalar', 'aggregate', 'pointer'] in OpenMP clause 'defaultmap'}}
  foo();
  #pragma omp target defaultmap (tofrom scalar) // expected-warning {{missing ':' after defaultmap modifier - ignoring}}
  foo();
  #pragma omp target defaultmap (tofrom, // expected-error {{expected ')'}} expected-error {{expected ['scalar', 'aggregate', 'pointer'] in OpenMP clause 'defaultmap'}} expected-warning {{missing ':' after defaultmap modifier - ignoring}} expected-note {{to match this '('}}
  foo();
  #pragma omp target defaultmap (scalar: // expected-error {{expected ')'}} expected-error {{expected ['alloc', 'from', 'to', 'tofrom', 'firstprivate', 'none', 'default'] in OpenMP clause 'defaultmap'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target defaultmap (tofrom, scalar // expected-error {{expected ')'}} expected-warning {{missing ':' after defaultmap modifier - ignoring}} expected-error {{expected ['scalar', 'aggregate', 'pointer'] in OpenMP clause 'defaultmap'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target defaultmap(tofrom:scalar) defaultmap(to:scalar) // expected-error {{at most one defaultmap clause for each variable-category can appear on the directive}}
  foo();
  #pragma omp target defaultmap(alloc:pointer) defaultmap(to:scalar) defaultmap(firstprivate:pointer) // expected-error {{at most one defaultmap clause for each variable-category can appear on the directive}}
  foo();
  #pragma omp target defaultmap(none:pointer) // expected-note {{explicit data sharing attribute, data mapping attribute, or is_devise_ptr clause requested here}}
  g++; // expected-error {{variable 'g' must have explicitly specified data sharing attributes, data mapping attributes, or in an is_device_ptr clause}}
  #pragma omp target defaultmap(none:pointer) defaultmap(none:scalar) map(argc) // expected-note {{explicit data sharing attribute, data mapping attribute, or is_devise_ptr clause requested here}}
  argc = *g; // expected-error {{variable 'g' must have explicitly specified data sharing attributes, data mapping attributes, or in an is_device_ptr clause}}
  #pragma omp target defaultmap(none:pointer) defaultmap(none:scalar) map(g) // expected-note {{explicit data sharing attribute, data mapping attribute, or is_devise_ptr clause requested here}}
  argc = *g; // expected-error {{variable 'argc' must have explicitly specified data sharing attributes, data mapping attributes, or in an is_device_ptr clause}}
  #pragma omp target defaultmap(none:pointer) defaultmap(none:scalar) defaultmap(none:aggregate) map(g) firstprivate(argc) // expected-note {{explicit data sharing attribute, data mapping attribute, or is_devise_ptr clause requested here}}
  argc = *g + arr[1]; // expected-error {{variable 'arr' must have explicitly specified data sharing attributes, data mapping attributes, or in an is_device_ptr clause}}
  #pragma omp target defaultmap(none:aggregate) defaultmap(none:scalar) map(argc) // expected-note {{explicit data sharing attribute, data mapping attribute, or is_devise_ptr clause requested here}}
  bar.a += argc; // expected-error {{variable 'bar' must have explicitly specified data sharing attributes, data mapping attributes, or in an is_device_ptr clause}}
  #pragma omp target defaultmap(none:aggregate) defaultmap(none:scalar) map(argc) // expected-note {{explicit data sharing attribute, data mapping attribute, or is_devise_ptr clause requested here}}
  baz.bar.a += argc; // expected-error {{variable 'baz' must have explicitly specified data sharing attributes, data mapping attributes, or in an is_device_ptr clause}}
  int vla[argc];
  #pragma omp target defaultmap(none:aggregate) defaultmap(none:scalar) map(argc) // expected-note {{explicit data sharing attribute, data mapping attribute, or is_devise_ptr clause requested here}}
  vla[argc-1] += argc; // expected-error {{variable 'vla' must have explicitly specified data sharing attributes, data mapping attributes, or in an is_device_ptr clause}}

  #pragma omp target defaultmap(none:aggregate) // expected-note {{explicit data sharing attribute, data mapping attribute, or is_devise_ptr clause requested here}}
  #pragma omp parallel
  baz.bar.a += argc; // expected-error {{variable 'baz' must have explicitly specified data sharing attributes, data mapping attributes, or in an is_device_ptr clause}}

  #pragma omp parallel
  #pragma omp target defaultmap(none:aggregate) // expected-note {{explicit data sharing attribute, data mapping attribute, or is_devise_ptr clause requested here}}
  #pragma omp parallel
  baz.bar.a += argc; // expected-error {{variable 'baz' must have explicitly specified data sharing attributes, data mapping attributes, or in an is_device_ptr clause}}

  #pragma omp parallel
  #pragma omp target defaultmap(none:aggregate) // expected-note {{explicit data sharing attribute, data mapping attribute, or is_devise_ptr clause requested here}}
  #pragma omp parallel
  *baz.p += argc; // expected-error {{variable 'baz' must have explicitly specified data sharing attributes, data mapping attributes, or in an is_device_ptr clause}}

  #pragma omp parallel
  #pragma omp target defaultmap(none:pointer) // expected-note {{explicit data sharing attribute, data mapping attribute, or is_devise_ptr clause requested here}}
  #pragma omp parallel
  bazPtr->p += argc; // expected-error {{variable 'bazPtr' must have explicitly specified data sharing attributes, data mapping attributes, or in an is_device_ptr clause}}

  return argc;
}

int main(int argc, char **argv) {
  #pragma omp target defaultmap // expected-error {{expected '(' after 'defaultmap'}}
  foo();
  #pragma omp target defaultmap ( // expected-error {{expected ['alloc', 'from', 'to', 'tofrom', 'firstprivate', 'none', 'default'] in OpenMP clause 'defaultmap'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target defaultmap () // expected-error {{expected ['alloc', 'from', 'to', 'tofrom', 'firstprivate', 'none', 'default'] in OpenMP clause 'defaultmap'}}
  foo();
  #pragma omp target defaultmap (tofrom // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-warning {{missing ':' after defaultmap modifier - ignoring}} expected-error {{expected ['scalar', 'aggregate', 'pointer'] in OpenMP clause 'defaultmap'}}
  foo();
  #pragma omp target defaultmap (tofrom: // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expected ['scalar', 'aggregate', 'pointer'] in OpenMP clause 'defaultmap'}}
  foo();
  #pragma omp target defaultmap (tofrom) // expected-warning {{missing ':' after defaultmap modifier - ignoring}} expected-error {{expected ['scalar', 'aggregate', 'pointer'] in OpenMP clause 'defaultmap'}}
  foo();
  #pragma omp target defaultmap (tofrom scalar) // expected-warning {{missing ':' after defaultmap modifier - ignoring}}
  foo();
  #pragma omp target defaultmap (tofrom, // expected-error {{expected ')'}} expected-error {{expected ['scalar', 'aggregate', 'pointer'] in OpenMP clause 'defaultmap'}} expected-warning {{missing ':' after defaultmap modifier - ignoring}} expected-note {{to match this '('}}
  foo();
  #pragma omp target defaultmap (scalar: // expected-error {{expected ')'}} expected-error {{expected ['alloc', 'from', 'to', 'tofrom', 'firstprivate', 'none', 'default'] in OpenMP clause 'defaultmap'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target defaultmap (tofrom, scalar // expected-error {{expected ')'}} expected-warning {{missing ':' after defaultmap modifier - ignoring}} expected-error {{expected ['scalar', 'aggregate', 'pointer'] in OpenMP clause 'defaultmap'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target defaultmap(tofrom:scalar) defaultmap(to:scalar) // expected-error {{at most one defaultmap clause for each variable-category can appear on the directive}}
  foo();
  #pragma omp target defaultmap(alloc:pointer) defaultmap(to:scalar) defaultmap(firstprivate:pointer) // expected-error {{at most one defaultmap clause for each variable-category can appear on the directive}}
  foo();
  #pragma omp target defaultmap(none:pointer) // expected-note {{explicit data sharing attribute, data mapping attribute, or is_devise_ptr clause requested here}}
  g++; // expected-error {{variable 'g' must have explicitly specified data sharing attributes, data mapping attributes, or in an is_device_ptr clause}}
  #pragma omp target defaultmap(none:pointer) defaultmap(none:scalar) map(argc) // expected-note {{explicit data sharing attribute, data mapping attribute, or is_devise_ptr clause requested here}}
  argc = *g; // expected-error {{variable 'g' must have explicitly specified data sharing attributes, data mapping attributes, or in an is_device_ptr clause}}
  #pragma omp target defaultmap(none:pointer) defaultmap(none:scalar) map(g) // expected-note {{explicit data sharing attribute, data mapping attribute, or is_devise_ptr clause requested here}}
  argc = *g; // expected-error {{variable 'argc' must have explicitly specified data sharing attributes, data mapping attributes, or in an is_device_ptr clause}}
  #pragma omp target defaultmap(none:pointer) defaultmap(none:scalar) defaultmap(none:aggregate) map(g) firstprivate(argc) // expected-note {{explicit data sharing attribute, data mapping attribute, or is_devise_ptr clause requested here}}
  argc = *g + arr[1]; // expected-error {{variable 'arr' must have explicitly specified data sharing attributes, data mapping attributes, or in an is_device_ptr clause}}
  #pragma omp target defaultmap(none:aggregate) defaultmap(none:scalar) map(argc) // expected-note {{explicit data sharing attribute, data mapping attribute, or is_devise_ptr clause requested here}}
  bar.a += argc; // expected-error {{variable 'bar' must have explicitly specified data sharing attributes, data mapping attributes, or in an is_device_ptr clause}}
  #pragma omp target defaultmap(none:aggregate) defaultmap(none:scalar) map(argc) // expected-note {{explicit data sharing attribute, data mapping attribute, or is_devise_ptr clause requested here}}
  baz.bar.a += argc; // expected-error {{variable 'baz' must have explicitly specified data sharing attributes, data mapping attributes, or in an is_device_ptr clause}}
  int vla[argc];
  #pragma omp target defaultmap(none:aggregate) defaultmap(none:scalar) map(argc) // expected-note {{explicit data sharing attribute, data mapping attribute, or is_devise_ptr clause requested here}}
  vla[argc-1] += argc; // expected-error {{variable 'vla' must have explicitly specified data sharing attributes, data mapping attributes, or in an is_device_ptr clause}}

  #pragma omp target defaultmap(none:aggregate) // expected-note {{explicit data sharing attribute, data mapping attribute, or is_devise_ptr clause requested here}}
  #pragma omp parallel
  baz.bar.a += argc; // expected-error {{variable 'baz' must have explicitly specified data sharing attributes, data mapping attributes, or in an is_device_ptr clause}}

  #pragma omp parallel
  #pragma omp target defaultmap(none:aggregate) // expected-note {{explicit data sharing attribute, data mapping attribute, or is_devise_ptr clause requested here}}
  #pragma omp parallel
  baz.bar.a += argc; // expected-error {{variable 'baz' must have explicitly specified data sharing attributes, data mapping attributes, or in an is_device_ptr clause}}

  #pragma omp parallel
  #pragma omp target defaultmap(none:aggregate) // expected-note {{explicit data sharing attribute, data mapping attribute, or is_devise_ptr clause requested here}}
  #pragma omp parallel
  *baz.p += argc; // expected-error {{variable 'baz' must have explicitly specified data sharing attributes, data mapping attributes, or in an is_device_ptr clause}}

  #pragma omp parallel
  #pragma omp target defaultmap(none:pointer) // expected-note {{explicit data sharing attribute, data mapping attribute, or is_devise_ptr clause requested here}}
  #pragma omp parallel
  bazPtr->p += argc; // expected-error {{variable 'bazPtr' must have explicitly specified data sharing attributes, data mapping attributes, or in an is_device_ptr clause}}

  return tmain<int, char, 1, 0>(argc, argv);  // expected-note {{in instantiation of function template specialization 'tmain<int, char, 1, 0>' requested here}}
}

