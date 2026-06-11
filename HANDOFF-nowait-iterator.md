# Handoff: `nowait` + iterator-modifier map/motion lowering (PR #199101)

> **Audience:** the engineer who will run the offload/GPU tests on a machine with an
> AMD GPU (`amdgpu`), `flang`, and `libomptarget`. This document captures the bug,
> the fix, every test that was added, the reasoning behind each decision, what could
> **not** be verified locally (no device here), and the exact commands to run.
>
> **Date:** 2026-06-11 · **Branch:** `users/cchen/llvmir_openmp_motion_map_iterator`
> (also mirrored as `wip_llvmir_openmp_motion_map_iterator`).

---

## 0. TL;DR — what the GPU machine must do

1. Build `flang`, `openmp`/`offload` (`libomptarget`) with the `amdgpu` plugin.
2. Run the single end-to-end offload test that exercises this work:

   ```bash
   # from the build dir that configured the offload test suite
   ./bin/llvm-lit -sv \
     <monorepo>/offload/test/offloading/fortran/map-motion-iterator.f90
   ```

   It is gated by `! REQUIRES: flang, amdgpu` and driven by
   `! RUN: %libomptarget-compile-fortran-run-and-check-generic`, so it compiles the
   Fortran, runs it on the device, and FileCheck-matches stdout.

3. **The cases that this PR's fix specifically makes correct are `f`, `g`, `h`
   (the `nowait` cases) and the stress cases `r/t`, `u/v`.** If those print the
   expected lines (Section 6), the fix is validated on real hardware. Everything
   else (`a`–`e`) is pre-existing coverage that should keep passing.

The IR-level and unit tests (Sections 7–8) already pass locally and need no device.

---

## 1. Branch / commit structure

```
d9aa82043acd  [OpenMP][mlir] Test declare mapper with iterator modifier on all map/motion ops   <- HEAD
b856c6e9d466  [OpenMP][mlir] Stress test iterator modifier with clause combinations
e07c929b9d63  [OpenMP][mlir] Support iterator modifier LLVM lowering for map/motion              <- the PR commit
dc732c735c29  Preserve iterator-dependent locators in shared map lowering                        <- base (Flang PR)
```

- **`e07c929b9d63`** is the actual PR #199101 commit. The `nowait` fix described in
  Sections 3–5 was developed as three reviewed commits and then **squashed into this
  single commit** (LLVM squash-merges anyway; the tree was verified byte-identical to
  the pre-squash stack). Its message documents the heap-array design + a before/after
  IR example.
- **`b856c6e9d466`** and **`d9aa82043acd`** are additive **test-only** commits on top,
  kept separate (per the author's request — not squashed into the PR base).
- Files touched across the whole stack (`git diff --stat dc732c735c29 HEAD`):

  | File | +/- | What |
  |---|---|---|
  | `llvm/lib/Frontend/OpenMP/OMPIRBuilder.cpp` | +336 | heap alloc/free + skip-privatize |
  | `llvm/include/llvm/Frontend/OpenMP/OMPIRBuilder.h` | ±54 | `emitTargetTask` new param |
  | `llvm/unittests/Frontend/OpenMPIRBuilderTest.cpp` | +266 | unit test |
  | `mlir/.../OpenMPToLLVMIRTranslation.cpp` | +165 | iterator translation (verify-only for nowait) |
  | `mlir/test/Target/LLVMIR/openmp-iterator.mlir` | +591 | lit tests |
  | `mlir/test/Target/LLVMIR/openmp-todo.mlir` | ±102 | negative iterated-depend TODO |
  | `offload/test/offloading/fortran/map-motion-iterator.f90` | +250 | runtime test |

> **Important:** the working tree also contains an **untracked** `.github/agents/`
> directory (review agent definitions). Keep it untracked — do **not** commit it.
> This handoff file (`HANDOFF-nowait-iterator.md`) is also untracked; copy it to the
> GPU machine separately (e.g. `scp`) since it will not travel with a `git push`.

---

## 2. Background: what the iterator modifier does in this layer

The iterator modifier (`map(iterator(i=0:n:1), to: a(i))`,
`from(iterator(i=...): x(i))`, etc.) lets a single map/motion clause expand into a
**runtime-determined number of entries**. In MLIR this is an `omp.iterator` region
producing `!omp.iterated<...>` operands consumed by `map_iterated(...)` on
`omp.target_data` / `target_enter_data` / `target_exit_data` / `target_update`.

During MLIR→LLVM-IR translation
(`mlir/lib/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.cpp`,
`convertOmpTargetData`), the iterator becomes an `IteratorInfo` + a `dynMapEntriesCB`
callback that **fills runtime-sized `.offload_*` arrays** inside a generated loop. The
runtime mapper call (`__tgt_target_data_{begin,end,update}_mapper`) is then passed the
runtime trip count as its entry count.

`Info.TotalMapCount` (an LLVM `Value*`) is the total runtime entry count;
`Info.HasNoWait` reflects the `nowait` clause.

---

## 3. The bug that was fixed (`nowait` + iterator truncation)

**Symptom (latent, silent):** a *standalone* `target enter/exit data nowait` or
`target update nowait` carrying an iterator modifier could transfer **garbage / wrong
data**, or read out of bounds, on the device.

**Root cause:** for `nowait`, the motion becomes a **deferred target task**. The old
code path (`emitTargetTask`) *privatized* the offload descriptor arrays
(`.offload_baseptrs`, `.offload_ptrs`, `.offload_sizes`, `.offload_maptypes`,
`.offload_mappers`, optionally `.offload_mapnames`) by copying them into a
**compile-time-sized** `struct.task_with_privates` member and `memcpy`-ing only
`getTypeStoreSize(elementType)` = **8 bytes (one element)** — while the runtime mapper
call was still handed `PointerNum = TotalMapCount` (the full runtime trip count).

So the deferred task read `TotalMapCount` entries out of a struct that only held one
→ **out-of-bounds read / silent wrong device transfers**. Because the offload arrays
are *element-typed dynamic allocas* (`alloca ptr, i64 %count`) regardless of whether
the trip count is constant, even a constant-trip iterator (`iterator(i=1:4)`) was
affected.

**Code sites involved (pre-fix):**
- element-typed dynamic allocas: `emitDynamicOffloadingArraysAllocas`
- privatization truncation: `getOffloadingArrayType` + the `memcpy storeSize(ElementType)`
- `nowait` was **unguarded** for motion ops (no TODO/diagnostic existed to catch it).

---

## 4. The fix — "Approach A": heap-allocate + capture pointer + free in task body

Gated strictly on **`Info.HasNoWait && Info.TotalMapCount != nullptr`** (the iterator
dynamic-array feature being active under `nowait`). The static / non-`nowait` paths
are byte-identical to before.

1. **Heap-allocate the offload arrays** with `__kmpc_alloc(gtid, count*8, /*allocator=*/null)`
   instead of a stack `alloca`, so the buffers outlive the generating stack frame.
   - `llvm/lib/Frontend/OpenMP/OMPIRBuilder.cpp`, `emitDynamicOffloadingArraysAllocas`:
     `bool UseHeap = Info.HasNoWait;` (≈ L10595), the `allocOffloadArray` lambda
     (≈ L10620) emits `__kmpc_alloc` sized `CreateMul(CreateIntCast(TotalMapCount,
     SizeTy), getTypeAllocSize(ElemTy))`. Element size is 8 (ptr / i64) for all arrays.
2. **Fill them as before** (`emitOffloadingArrays` + the iterator loop + `dynMapEntriesCB`
   write through `Info.RTArgs.*`). This FILL happens in `BeginThenGen` **before** the
   task spawns — heap pointers are defined+filled pre-spawn, read post-spawn ⇒ clean
   live-ins.
3. **Do NOT privatize** the arrays. New trailing param
   `bool RuntimeLivedOffloadArrays` on `emitTargetTask`
   (`OMPIRBuilder.h` ≈ L2982, default `false`; `OMPIRBuilder.cpp` ≈ L9288). When true,
   the `OffloadingArraysToPrivatize` population loop is skipped (≈ L9455
   `if (NeedsTargetTask && !RuntimeLivedOffloadArrays)`), so the 8-byte heap **pointers**
   are captured as ordinary `kmp_task_t` shareds via the normal `CodeExtractor`
   aggregate.
4. **Free in the deferred task body** with `__kmpc_free(gtid, ptr, null)` for each
   allocated array, after the mapper runtime call
   (`OMPIRBuilder.cpp` ≈ L8560 `RuntimeLivedOffloadArrays = Info.HasNoWait &&
   Info.TotalMapCount != nullptr;`, frees at ≈ L8587, guarded `Arr && !isa<Constant>(Arr)`).
5. **Defense-in-depth assert** at the with-body `target data` preallocation site
   (≈ L8509 `assert(!Info.HasNoWait && ...)`), since `target data` (with a body) has no
   `nowait` and must never reach the heap path.

**Why each offload array is freed exactly once / only-if-allocated:** the free
predicate mirrors each array's allocation predicate. `MappersArray` is allocated even
when there is no user mapper; `MapNamesArray` is null when debug info is off
(`Arr && !isa<Constant>(Arr)` skips the null/constant ones). Standalone motion ops are
single-call, so `MapTypesArrayEnd` (only the two-call `target data` shape, which has no
`nowait`) is never involved.

---

## 5. Why Approach A is correct (and spec-conformant)

- **Not a spec violation to capture heap pointers as `kmp_task_t` "shareds".** The
  OpenMP spec governs *observable semantics*, not the internal representation of the
  offload descriptor arrays or the `kmp_task_t` packaging. "shareds" here is a *runtime
  implementation* term (a snapshot block `memcpy`'d into the task at creation), **not**
  the OpenMP `shared`/`private` data-sharing clauses. Capturing the heap *pointer
  values* gives the deferred task a stable snapshot; the heap buffer is logically owned
  by the task after creation.
- **Observable contract preserved.** For `target enter/exit data` / `target update`
  + `nowait`, the spec requires a deferred target task ordered by
  `taskwait`/barrier/`depend`. Array FILL runs pre-spawn; only the mapper call runs in
  the deferred body.
- **Cross-thread free is legal.** A deferred task may run on a different thread than the
  allocator; OpenMP allows `omp_free` from a different thread for non-thread-private
  allocators. `__kmpc_alloc`/`__kmpc_free` with the **default (null)** allocator is
  conformant (and chosen deliberately over libc `malloc`/`free` for runtime
  consistency). **Do not** switch to a thread-private allocator — that would make the
  cross-thread free UB.
- **Cleaner SSA.** Heap pointers are plain call results, sidestepping the
  CodeExtractor-with-allocas awkwardness that motivated the original privatize copy.

Approach A was chosen over a "runtime-sized task tail" (Approach B) because it is
localized: no proxy/offset machinery changes, reuses the existing shareds capture.

---

## 6. The offload runtime test — cases, expected output, and rationale

File: `offload/test/offloading/fortran/map-motion-iterator.f90`
(`n = 8`; arrays initialized `x(i) = i`). Each `nowait` motion is made observable on
the host with an immediately following `!$omp taskwait`. Expected stdout lines (these
are the `! CHECK:` lines — FileCheck verifies them):

| Case | Construct exercised | Expected line |
|---|---|---|
| a | `target update from(iterator)` | `update from: 1 102 3 104 5 106 7 108` |
| b | `target update to/from(iterator)` | `update tofrom: 1 212 3 214 5 216 7 218` |
| c | `target exit data map(iterator, from:)` | `exit data: 11 2 33 4 55 6 77 8` |
| d | `target data map(iterator, tofrom:)` | `target data: 21 2 43 4 65 6 87 8` |
| e | `target data if(...) map(iterator)` (dynamic bounds) | `target data if: 31 2 53 4 75 6 97 8` |
| **f** | **`target update from(iterator) nowait`** | `update from nowait: 1 302 3 304 5 306 7 308` |
| **g** | **`target enter data` + `update to/from(iterator) nowait`** | `enter/update nowait: 402 2 404 4 406 6 408 8` |
| **h** | **`target exit data map(iterator, from:) nowait`** | `exit data nowait: 501 2 503 4 505 6 507 8` |
| **r** | stress: whole-array static + iterator element map together | `mixed static+iter r: 801 802 803 804 805 806 807 808` |
| **t** | (same directive as r, the iterator half) | `mixed static+iter t: 91 2 93 4 95 6 97 8` |
| **u** | stress: two disjoint motion iterators in one `update` (odd) | `multi iter u: 601 2 603 4 605 6 607 8` |
| **v** | (same directive as u, even) | `multi iter v: 1 702 3 704 5 706 7 708` |

**Cases f/g/h are the fix's direct targets.** They are the ones that, before the fix,
could print wrong values or read OOB.

### Rationale notes worth knowing while debugging on device

- **Why `!$omp taskwait` after every `nowait` motion:** the deferred target task is
  otherwise unordered w.r.t. the host read. The `taskwait` is what makes the result
  observable and the test deterministic.
- **The `map(present, alloc:)` per-element pattern (cases d, e, g, r):** when a device
  region touches array elements that were brought over by an **iterator element map**
  (e.g. only odd elements), you must map each element explicitly with
  `map(present, alloc: g(1)) ... g(7)`. **Do not** use a plain `!$omp target` over the
  whole array — that emits an *implicit whole-array map*, and `libomptarget`'s
  `lookupMapping`/`getTargetPointer` resolve it to the **first** element's device buffer
  (an `ExtendsAfter && IsImplicit` reuse), so writes to `g(3/5/7)` land at
  `base + 8/16/24` **past** the 4-byte allocation of the `g(1)` entry → device OOB and
  the other per-element buffers never get updated. This was a real bug caught in review
  for case `g` and fixed by switching to explicit per-element present maps (mirroring
  the pre-existing cases d/e). **If case `g` ever prints `402 2 403 4 405 6 407 8`
  instead of the expected `...404...406...408...`, this implicit-whole-array trap has
  regressed.**
- **Arithmetic sanity (so you can eyeball failures):**
  - f: device sets even elements to `300+i` (302/304/306/308); host clobbers them to
    −100; `update from(iterator even)` pulls the device values back → odds stay
    `1 3 5 7`, evens become `302 304 306 308`.
  - g: enter-data `to` odds, host sets odds `400+i`, `update to` odds → device, device
    `+1` (401→402, etc.), host clobbers odds to −100, `update from` odds → `402 404 406
    408` on odds, evens untouched `2 4 6 8`.
  - h: device sets odds `500+i`, host clobbers odds −100, exit-data `from(iterator odd)`
    pulls device → `501 503 505 507` odds, evens `2 4 6 8`.
  - r/t, u/v: see the `! CHECK:` lines above; values chosen distinct per element so a
    misrouted entry shows up immediately.

### If you cannot run on device

The IR/unit tests (Sections 7–8) structurally verify the heap alloc/free, the
non-privatized capture, and the runtime entry count. They were **all run and pass
locally**. The runtime values themselves (Section 6 table) are the part that requires
the GPU.

---

## 7. MLIR → LLVM-IR lit tests (no device needed; already passing)

File: `mlir/test/Target/LLVMIR/openmp-iterator.mlir` — a `split-file` test with a
`host.mlir` part (`CHECK` prefix) and a `target.mlir` part (`TARGET` prefix). The
`target.mlir` module carries `omp.is_target_device = false,
omp.target_triples = ["amdgcn-amd-amdhsa"]`.

Run both relevant lit files:

```bash
./build/bin/llvm-lit -sv \
  mlir/test/Target/LLVMIR/openmp-iterator.mlir \
  mlir/test/Target/LLVMIR/openmp-todo.mlir
```

Relevant functions / CHECK groups added by this work:

- **`nowait` heap arrays** — `target_update_map_iterator_nowait`,
  `target_enter_data_map_iterator_nowait`: assert `call ptr @__kmpc_alloc`,
  `@__kmpc_omp_target_task_alloc`, `.omp_target_task_proxy_func`,
  `@__kmpc_free`, the mapper call's count operand = the runtime trip count (not a
  constant 1), and a bounded `TARGET-NOT` after `TARGET-COUNT-7` to catch a
  double/over-free.
- **Stress: clause combinations** (commit `b856c6e9d466`) —
  `target_data_iterator_use_device_ptr` (asserts the offload array sizing accounts for
  both the iterated entries and the `use_device_ptr` return-param entry),
  `_use_device_addr`, `target_data/.../target_update`+`if`+`device`.
- **Declare mapper + iterator** (commit `d9aa82043acd`) —
  `target_enter_data_map_iterator_mapper` (pre-existing) plus the new
  `target_update_map_iterator_mapper`, `target_exit_data_map_iterator_mapper`,
  `target_data_map_iterator_mapper`. Each asserts `alloca ptr, i64 3` for
  `.offload_mappers`, the per-iteration
  `getelementptr ... store ptr @.omp_mapper.simple_mapper` into that array, and the
  `.offload_mappers` operand reaching `__tgt_target_data_{update,end,begin}_mapper`
  with an `i32 3` count. The `TARGET-LABEL` order **must** match the IR emission order
  (`enter → update → exit → data`).

**Negative test:** `mlir/test/Target/LLVMIR/openmp-todo.mlir`,
`target_enter_data_iterated_depend` (≈ L239) expects
`not yet implemented: Unhandled clause depend in omp.target_enter_data operation` —
i.e. an **iterated `depend`** on the motion ops is a clean TODO, not a crash.

---

## 8. Unit test (no device needed; already passing)

`llvm/unittests/Frontend/OpenMPIRBuilderTest.cpp`,
`TEST_F(OpenMPIRBuilderTest, TargetDataNoWaitDynamicHeapArrays)` (≈ L8263).

```bash
ninja -C build OpenMPIRBuilderTests   # or LLVMFrontendTests
./build/.../OpenMPIRBuilderTests --gtest_filter='*TargetDataNoWaitDynamicHeapArrays*'
# whole suite must stay green:
./build/.../OpenMPIRBuilderTests   # 103/103 locally
```

Key construction details (these are easy to get wrong if you edit the test):
- Uses **`SeparateBeginEndCalls = true`** to mirror the 7-array production shape.
- **`Info.TotalMapCount` must be NON-constant** (a function-arg `i64`). A constant routes
  to `AllocaIP` and *hides* the heap path entirely → the test would silently verify the
  wrong path.
- `Info.HasNoWait = true`.
- Must call **`OMPBuilder.finalize()` before inspecting IR**: the skip-privatize,
  `__kmpc_free`, proxy, and `__kmpc_omp_target_task_alloc` are emitted during outlining
  (PostOutlineCB), which runs in `finalize()`.
- Asserts: ≥ 6 balanced `__kmpc_alloc`/`__kmpc_free`, mapper-call count operand
  non-constant, and **no** truncated `struct.privates` array member.

> **Scope honesty:** this unit test verifies the IR structural contract only, **not**
> runtime data-transfer correctness. The latter is exactly what the Section 6 offload
> test (your job on the GPU box) covers.

---

## 9. Declare mapper + iterator modifier — supported on all four directives

Question that was investigated: *is a user-defined `declare mapper` combined with the
iterator modifier supported, or should it trigger a "not yet implemented" check?*

**Answer: fully supported on all four directives.** During translation,
`getOrCreateUserDefinedMapperFunc` is called per iteration and the mapper function
pointer is stored into `.offload_mappers`, which flows to
`__tgt_target_data_{begin,end,update}_mapper`. Empirically confirmed all four emit
`@.omp_mapper.<name>`. Previously only `target_enter_data` had a test; commit
`d9aa82043acd` adds the other three (Section 7). No guard/TODO was added because the
feature works.

---

## 10. Deferred / intentionally-not-done items

- **`checkDepend` robustness nit (NOT a live bug; author said "leave it"):** the three
  target-data motion ops reject *all* `depend` (plain and iterated) via a pre-existing
  `checkDepend` TODO. It keys off `op.getDependKinds()` presence. The parser sets
  `depend_kinds = []` (present-but-empty) for iterated-only depend, so the TODO **does**
  fire for iterated depend (it is not silently dropped) — verified by the negative test
  in Section 7. A directly-built op with *only* `depend_iterated` and no `depend_kinds`
  would slip through; a cleaner guard would also check
  `!op.getDependIterated().empty()`. Left as-is by request.
- **No device/`use_device_*` Fortran runtime cases** were added — their values are not
  deterministically verifiable without a device, and `omp_lib`/`iso_c_binding.mod` were
  unavailable in the dev environment. They are covered structurally at the MLIR level
  instead (Section 7).

---

## 11. CI/CD failure investigation (read before blaming this PR)

CI was investigated twice. **The failures are not caused by this PR.**

- **Root cause:** a pre-existing PowerPC test,
  `llvm/test/CodeGen/PowerPC/fast-isel-cmp-imm.ll` (an SPE check), is broken at the
  branch's base by upstream SPE feature commit `4d0100789dc9` + its revert
  `c24ab4c814f6` (both already in the base) interacting with a GlobalISel change. **This
  PR touches zero PowerPC/CodeGen files.**
- The GitHub Actions Linux jobs additionally hit pure infrastructure failures (runner
  communication loss / exit 137 OOM), unrelated to any source change.
- **Recommended action:** rebase onto a newer `main` (so the PowerPC test is fixed
  upstream) and/or re-run the infra-flaky jobs. Nothing in this PR needs changing to fix
  CI.

---

## 12. Quick reference — key code locations (current line numbers, branch HEAD)

| Symbol / site | File | ≈ Line |
|---|---|---|
| `bool UseHeap = Info.HasNoWait;` | `llvm/lib/Frontend/OpenMP/OMPIRBuilder.cpp` | 10595 |
| `allocOffloadArray` lambda (`__kmpc_alloc`) | same | 10620 |
| array allocations (`BasePointersArray`, ...) | same | 10632–10638 |
| `RuntimeLivedOffloadArrays = HasNoWait && TotalMapCount` | same | 8560 |
| `__kmpc_free` in TaskBodyCB | same | 8587 |
| defense-in-depth `assert(!Info.HasNoWait ...)` | same | 8509 |
| `emitTargetTask(... bool RuntimeLivedOffloadArrays)` def | same | 9288 |
| skip-privatize guard `!RuntimeLivedOffloadArrays` | same | 9455 |
| `emitTargetTask` decl + doc (default `= false`) | `llvm/include/llvm/Frontend/OpenMP/OMPIRBuilder.h` | 2972, 2982 |
| `TargetDataNoWaitDynamicHeapArrays` unit test | `llvm/unittests/Frontend/OpenMPIRBuilderTest.cpp` | 8263 |
| iterated-depend negative TODO | `mlir/test/Target/LLVMIR/openmp-todo.mlir` | 239 |
| runtime offload test | `offload/test/offloading/fortran/map-motion-iterator.f90` | — |

Runtime functions involved: `OMPRTL___kmpc_alloc` / `OMPRTL___kmpc_free`
(`llvm/include/llvm/Frontend/OpenMP/OMPKinds.def`; args `gtid:i32, size:size_t,
allocator:VoidPtr=null`), `__kmpc_omp_target_task_alloc`,
`__tgt_target_data_{begin,end,update}_mapper`.

---

## 13. Local environment caveats (why device tests were not run here)

- No AMD device / no `libomptarget` / no `iso_c_binding.mod` in the dev box → the
  Fortran offload test (`REQUIRES: flang, amdgpu`) **cannot run locally**; it was
  validated only by lowering + reasoning. **This is the gap the GPU machine closes.**
- System `split-file` / `rg` were not installed; use the in-tree
  `./build/bin/split-file` and the editor's search tools.
- Tooling used locally for IR inspection:
  `./build/bin/split-file <test> /tmp/out` then
  `./build/bin/mlir-translate --mlir-to-llvmir /tmp/out/target.mlir`.

---

## 14. Suggested validation checklist on the GPU machine

1. `git log --oneline -4` shows the three PR commits (`d9aa82043acd`,
   `b856c6e9d466`, `e07c929b9d63`) on top of the base `dc732c735c29` from Section 1.
2. Build flang + offload (amdgpu) + run:
   `./bin/llvm-lit -sv .../offload/test/offloading/fortran/map-motion-iterator.f90` →
   **PASS** (all 12 `CHECK` lines, Section 6).
3. (Sanity, no device) `./build/bin/llvm-lit -sv
   mlir/test/Target/LLVMIR/openmp-iterator.mlir
   mlir/test/Target/LLVMIR/openmp-todo.mlir` → 2/2 PASS.
4. (Sanity, no device) `OpenMPIRBuilderTests` → 103/103 PASS.
5. If case `g` prints `402 2 403 4 405 6 407 8` (wrong) → the implicit-whole-array trap
   regressed (Section 6 rationale). If f/g/h print garbage or the process faults → the
   `nowait` heap-array fix regressed (Sections 3–4).
