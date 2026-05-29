// RUN: split-file %s %t
// RUN: mlir-translate -mlir-to-llvmir %t/host-only.mlir | FileCheck %s --check-prefix=HOST

//--- host-only.mlir

module attributes {omp.is_target_device = false} {
  llvm.func @host_only_capture(%addr : !llvm.ptr) {
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %c4 = llvm.mlir.constant(4 : i64) : i64
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %it = omp.iterator(%iv: i64) = (%c0 to %c4 step %c1) {
      %map = omp.map.info var_ptr(%addr : !llvm.ptr, i32)
        map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = ""}
      omp.yield(%map : !llvm.ptr)
    } -> !omp.iterated<!llvm.ptr>
    omp.target map_iterated(%it : !omp.iterated<!llvm.ptr>)
        map_iterated_captures(%addr -> %arg0 : !llvm.ptr) {
      %c42 = llvm.mlir.constant(42 : i32) : i32
      llvm.store %c42, %arg0 : i32, !llvm.ptr
      omp.terminator
    }
    llvm.return
  }

  llvm.func @host_only_mapped_capture(%addr : !llvm.ptr) {
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %c4 = llvm.mlir.constant(4 : i64) : i64
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %static = omp.map.info var_ptr(%addr : !llvm.ptr, i32)
      map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = ""}
    %it = omp.iterator(%iv: i64) = (%c0 to %c4 step %c1) {
      %map = omp.map.info var_ptr(%addr : !llvm.ptr, i32)
        map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = ""}
      omp.yield(%map : !llvm.ptr)
    } -> !omp.iterated<!llvm.ptr>
    omp.target map_iterated(%it : !omp.iterated<!llvm.ptr>)
        map_entries(%static -> %map_arg : !llvm.ptr)
        map_iterated_captures(%addr -> %capture_arg : !llvm.ptr) {
      %c7 = llvm.mlir.constant(7 : i32) : i32
      llvm.store %c7, %capture_arg : i32, !llvm.ptr
      omp.terminator
    }
    llvm.return
  }
}

// HOST-LABEL: define void @host_only_capture
// HOST-SAME: (ptr %[[ADDR:.*]]) {
// HOST-NOT: __tgt_target_kernel
// HOST: call void @[[OUTLINE_CAPTURE:[^(]+]](ptr %[[ADDR]], ptr null)

// HOST: define void @host_only_mapped_capture
// HOST-SAME: (ptr %[[ADDR2:.*]]) {
// HOST-NOT: __tgt_target_kernel
// HOST: call void @[[OUTLINE_MAPPED:[^(]+]](ptr %[[ADDR2]], ptr null)
// HOST: define internal void @[[OUTLINE_CAPTURE]](ptr %[[CAPTURE_ARG:.*]], ptr %{{.*}})
// HOST: store i32 42, ptr %[[CAPTURE_ARG]]
// HOST: define internal void @[[OUTLINE_MAPPED]](ptr %[[MAPPED_ARG:.*]], ptr %{{.*}})
// HOST: store i32 7, ptr %[[MAPPED_ARG]]
