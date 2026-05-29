// RUN: split-file %s %t
// RUN: mlir-translate -mlir-to-llvmir %t/host-only.mlir | FileCheck %s --check-prefix=HOST
// RUN: mlir-translate -mlir-to-llvmir %t/host-offload.mlir | FileCheck %s --check-prefix=OFFLOAD

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

  llvm.func @host_only_member_iterator(%addr : !llvm.ptr) {
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %c4 = llvm.mlir.constant(4 : i64) : i64
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %it = omp.iterator(%iv: i64) = (%c0 to %c4 step %c1) {
      %field = llvm.getelementptr %addr[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"map_type", (i32)>
      %field_map = omp.map.info var_ptr(%field : !llvm.ptr, i32)
        map_clauses(to) capture(ByRef) -> !llvm.ptr {name = ""}
      %map = omp.map.info var_ptr(%addr : !llvm.ptr, !llvm.struct<"map_type", (i32)>)
        map_clauses(to) capture(ByRef) members(%field_map : [0] : !llvm.ptr) -> !llvm.ptr {name = ""}
      omp.yield(%map : !llvm.ptr)
    } -> !omp.iterated<!llvm.ptr>
    omp.target map_iterated(%it : !omp.iterated<!llvm.ptr>)
        map_iterated_captures(%addr -> %arg0 : !llvm.ptr) {
      %field = llvm.getelementptr %arg0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"map_type", (i32)>
      %c11 = llvm.mlir.constant(11 : i32) : i32
      llvm.store %c11, %field : i32, !llvm.ptr
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
// HOST: define void @host_only_member_iterator
// HOST-SAME: (ptr %[[ADDR3:.*]]) {
// HOST-NOT: __tgt_target_kernel
// HOST: call void @[[OUTLINE_MEMBER:[^(]+]](ptr %[[ADDR3]], ptr null)
// HOST: define internal void @[[OUTLINE_CAPTURE]](ptr %[[CAPTURE_ARG:.*]], ptr %{{.*}})
// HOST: store i32 42, ptr %[[CAPTURE_ARG]]
// HOST: define internal void @[[OUTLINE_MAPPED]](ptr %[[MAPPED_ARG:.*]], ptr %{{.*}})
// HOST: store i32 7, ptr %[[MAPPED_ARG]]
// HOST: define internal void @[[OUTLINE_MEMBER]](ptr %[[MEMBER_ARG:.*]], ptr %{{.*}})
// HOST: %[[FIELD:.*]] = getelementptr
// HOST-SAME: ptr %[[MEMBER_ARG]]
// HOST: store i32 11, ptr %[[FIELD]]

//--- host-offload.mlir

module attributes {omp.is_target_device = false, omp.target_triples = ["amdgcn-amd-amdhsa"]} {
  llvm.func @target_map_iterator_capture(%addr : !llvm.ptr) {
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %c2 = llvm.mlir.constant(2 : i64) : i64
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %it = omp.iterator(%iv: i64) = (%c0 to %c2 step %c1) {
      %elem = llvm.getelementptr %addr[%iv] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %map = omp.map.info var_ptr(%elem : !llvm.ptr, i32)
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

  llvm.func @target_map_iterator_mapped_capture(%addr : !llvm.ptr) {
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %c2 = llvm.mlir.constant(2 : i64) : i64
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %static = omp.map.info var_ptr(%addr : !llvm.ptr, i32)
      map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = ""}
    %it = omp.iterator(%iv: i64) = (%c0 to %c2 step %c1) {
      %elem = llvm.getelementptr %addr[%iv] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %map = omp.map.info var_ptr(%elem : !llvm.ptr, i32)
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

  llvm.func @target_map_iterator_dynamic_capture(
      %addr : !llvm.ptr, %lb : i64, %ub : i64, %step : i64) {
    %it = omp.iterator(%iv: i64) = (%lb to %ub step %step) {
      %elem = llvm.getelementptr %addr[%iv] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %map = omp.map.info var_ptr(%elem : !llvm.ptr, i32)
        map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = ""}
      omp.yield(%map : !llvm.ptr)
    } -> !omp.iterated<!llvm.ptr>
    omp.target map_iterated(%it : !omp.iterated<!llvm.ptr>)
        map_iterated_captures(%addr -> %arg0 : !llvm.ptr) {
      %c9 = llvm.mlir.constant(9 : i32) : i32
      llvm.store %c9, %arg0 : i32, !llvm.ptr
      omp.terminator
    }
    llvm.return
  }
}

// OFFLOAD-LABEL: define void @target_map_iterator_capture
// OFFLOAD-SAME: (ptr %[[ADDR:[0-9]+]])
// OFFLOAD-DAG: %[[BASEPTRS:[^ ]*offload_baseptrs]] = alloca ptr, i64 5
// OFFLOAD-DAG: %[[TYPES:[^ ]*offload_maptypes]] = alloca i64, i64 5
// OFFLOAD: store ptr %[[ADDR]], ptr %{{.*}}
// OFFLOAD: store i64 32, ptr %{{.*}}
// OFFLOAD: store i64 288, ptr %{{.*}}
// OFFLOAD: omp_map_iterator.body:
// OFFLOAD: %[[ELEM:.*]] = getelementptr i32, ptr %[[ADDR]], i64 %{{.*}}
// OFFLOAD: %[[IDX:.*]] = add i64 2, %omp_map_iterator.iv
// OFFLOAD: %[[ITER_BP:.*]] = getelementptr inbounds ptr, ptr %[[BASEPTRS]], i64 %[[IDX]]
// OFFLOAD: store ptr %[[ELEM]], ptr %[[ITER_BP]]
// OFFLOAD: %[[ITER_TYPE:.*]] = getelementptr inbounds i64, ptr %[[TYPES]], i64 %[[IDX]]
// OFFLOAD: store i64 3, ptr %[[ITER_TYPE]]
// OFFLOAD: store i32 5, ptr %{{.*}}
// OFFLOAD: call i32 @__tgt_target_kernel
// OFFLOAD: call void @{{.*target_map_iterator_capture.*}}(ptr %[[ADDR]], ptr null)

// OFFLOAD-LABEL: define void @target_map_iterator_mapped_capture
// OFFLOAD-SAME: (ptr %[[ADDR2:[0-9]+]])
// OFFLOAD-DAG: %[[BASEPTRS2:[^ ]*offload_baseptrs]] = alloca ptr, i64 5
// OFFLOAD-DAG: %[[TYPES2:[^ ]*offload_maptypes]] = alloca i64, i64 5
// OFFLOAD: store ptr %[[ADDR2]], ptr %{{.*}}
// OFFLOAD: store i64 35, ptr %{{.*}}
// OFFLOAD: store i64 288, ptr %{{.*}}
// OFFLOAD: omp_map_iterator.body:
// OFFLOAD: %[[ELEM2:.*]] = getelementptr i32, ptr %[[ADDR2]], i64 %{{.*}}
// OFFLOAD: %[[IDX2:.*]] = add i64 2, %omp_map_iterator.iv
// OFFLOAD: %[[ITER_BP2:.*]] = getelementptr inbounds ptr, ptr %[[BASEPTRS2]], i64 %[[IDX2]]
// OFFLOAD: store ptr %[[ELEM2]], ptr %[[ITER_BP2]]
// OFFLOAD: %[[ITER_TYPE2:.*]] = getelementptr inbounds i64, ptr %[[TYPES2]], i64 %[[IDX2]]
// OFFLOAD: store i64 3, ptr %[[ITER_TYPE2]]
// OFFLOAD: store i32 5, ptr %{{.*}}
// OFFLOAD: call i32 @__tgt_target_kernel
// OFFLOAD: call void @{{.*target_map_iterator_mapped_capture.*}}(ptr %[[ADDR2]], ptr null)

// OFFLOAD-LABEL: define void @target_map_iterator_dynamic_capture
// OFFLOAD-SAME: (ptr %[[ADDR3:[0-9]+]], i64 %[[LB:[0-9]+]], i64 %[[UB:[0-9]+]], i64 %[[STEP:[0-9]+]])
// OFFLOAD: %[[DIFF:.*]] = sub i64 %[[UB]], %[[LB]]
// OFFLOAD: %[[DIV:.*]] = sdiv i64 %[[DIFF]], %[[STEP]]
// OFFLOAD: %[[TRIPS:.*]] = add i64 %[[DIV]], 1
// OFFLOAD: %[[SCALED:.*]] = mul i64 1, %[[TRIPS]]
// OFFLOAD: %[[TOTAL:.*]] = add i64 2, %[[SCALED]]
// OFFLOAD-DAG: %[[BASEPTRS3:[^ ]*offload_baseptrs]] = alloca ptr, i64 %[[TOTAL]]
// OFFLOAD-DAG: %[[TYPES3:[^ ]*offload_maptypes]] = alloca i64, i64 %[[TOTAL]]
// OFFLOAD: omp_map_iterator.body:
// OFFLOAD: %[[PHYS:.*]] = add i64 %[[LB]], %{{.*}}
// OFFLOAD: %[[ELEM3:.*]] = getelementptr i32, ptr %[[ADDR3]], i64 %[[PHYS]]
// OFFLOAD: %[[IDX3:.*]] = add i64 2, %omp_map_iterator.iv
// OFFLOAD: %[[ITER_BP3:.*]] = getelementptr inbounds ptr, ptr %[[BASEPTRS3]], i64 %[[IDX3]]
// OFFLOAD: store ptr %[[ELEM3]], ptr %[[ITER_BP3]]
// OFFLOAD: %[[ITER_TYPE3:.*]] = getelementptr inbounds i64, ptr %[[TYPES3]], i64 %[[IDX3]]
// OFFLOAD: store i64 3, ptr %[[ITER_TYPE3]]
// OFFLOAD: %[[NPTRS:.*]] = trunc i64 %[[TOTAL]] to i32
// OFFLOAD: store i32 %[[NPTRS]], ptr %{{.*}}
// OFFLOAD: call i32 @__tgt_target_kernel
// OFFLOAD: define internal void @{{.*target_map_iterator_capture.*}}(ptr %[[CAPTURE_ARG:.*]], ptr %{{.*}})
// OFFLOAD: store i32 42, ptr %[[CAPTURE_ARG]]
// OFFLOAD: define internal void @{{.*target_map_iterator_mapped_capture.*}}(ptr %[[MAPPED_ARG:.*]], ptr %{{.*}})
// OFFLOAD: store i32 7, ptr %[[MAPPED_ARG]]
