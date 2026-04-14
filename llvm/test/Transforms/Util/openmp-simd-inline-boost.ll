; RUN: opt -passes='openmp-simd-inline-boost' -S < %s | FileCheck %s

; Test that calls inside loops with llvm.loop.vectorize.enable metadata
; get the "function-inline-threshold" attribute boosted.

define void @callee(ptr %p) {
  %v = load float, ptr %p
  %r = fadd float %v, 1.0
  store float %r, ptr %p
  ret void
}

; CHECK-LABEL: define void @simd_loop_caller
define void @simd_loop_caller(ptr %a, i64 %n) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %ptr = getelementptr float, ptr %a, i64 %iv
; CHECK: call void @callee(ptr %ptr) #[[ATTR:[0-9]+]]
  call void @callee(ptr %ptr)
  %iv.next = add i64 %iv, 1
  %cmp = icmp slt i64 %iv.next, %n
  br i1 %cmp, label %loop, label %exit, !llvm.loop !0

exit:
  ret void
}

; Calls outside SIMD loops should NOT be boosted.
; CHECK-LABEL: define void @non_simd_loop_caller
define void @non_simd_loop_caller(ptr %a, i64 %n) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %ptr = getelementptr float, ptr %a, i64 %iv
; CHECK: call void @callee(ptr %ptr)
; CHECK-NOT: "function-inline-threshold"
  call void @callee(ptr %ptr)
  %iv.next = add i64 %iv, 1
  %cmp = icmp slt i64 %iv.next, %n
  br i1 %cmp, label %loop, label %exit, !llvm.loop !3

exit:
  ret void
}

; Calls outside any loop should NOT be boosted.
; CHECK-LABEL: define void @no_loop_caller
define void @no_loop_caller(ptr %p) {
; CHECK: call void @callee(ptr %p)
; CHECK-NOT: "function-inline-threshold"
  call void @callee(ptr %p)
  ret void
}

; Calls in nested loops where the outer loop is SIMD should be boosted.
; CHECK-LABEL: define void @nested_loop_caller
define void @nested_loop_caller(ptr %a, i64 %n, i64 %m) {
entry:
  br label %outer

outer:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  br label %inner

inner:
  %j = phi i64 [ 0, %outer ], [ %j.next, %inner ]
  %idx = mul i64 %i, %m
  %idx2 = add i64 %idx, %j
  %ptr = getelementptr float, ptr %a, i64 %idx2
; CHECK: call void @callee(ptr %ptr) #[[ATTR]]
  call void @callee(ptr %ptr)
  %j.next = add i64 %j, 1
  %jcmp = icmp slt i64 %j.next, %m
  br i1 %jcmp, label %inner, label %outer.latch

outer.latch:
  %i.next = add i64 %i, 1
  %icmp = icmp slt i64 %i.next, %n
  br i1 %icmp, label %outer, label %exit, !llvm.loop !0

exit:
  ret void
}

; Intrinsic calls should NOT be boosted.
; CHECK-LABEL: define void @intrinsic_in_simd
define void @intrinsic_in_simd(ptr %a, i64 %n) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %ptr = getelementptr float, ptr %a, i64 %iv
  %v = load float, ptr %ptr
; CHECK: %r = call float @llvm.sqrt.f32(float %v)
; CHECK-NOT: "function-inline-threshold"
  %r = call float @llvm.sqrt.f32(float %v)
  store float %r, ptr %ptr
  %iv.next = add i64 %iv, 1
  %cmp = icmp slt i64 %iv.next, %n
  br i1 %cmp, label %loop, label %exit, !llvm.loop !0

exit:
  ret void
}

; Calls that already have "function-inline-threshold" should NOT be overridden.
; CHECK-LABEL: define void @existing_threshold
define void @existing_threshold(ptr %a, i64 %n) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %ptr = getelementptr float, ptr %a, i64 %iv
; CHECK: call void @callee(ptr %ptr) #[[EXISTING:[0-9]+]]
  call void @callee(ptr %ptr) #0
  %iv.next = add i64 %iv, 1
  %cmp = icmp slt i64 %iv.next, %n
  br i1 %cmp, label %loop, label %exit, !llvm.loop !0

exit:
  ret void
}

declare float @llvm.sqrt.f32(float)

attributes #0 = { "function-inline-threshold"="500" }

; CHECK-DAG: attributes #[[ATTR]] = { "function-inline-threshold"="2000" }
; CHECK-DAG: attributes #[[EXISTING]] = { "function-inline-threshold"="500" }

; SIMD loop metadata
!0 = distinct !{!0, !1, !2}
!1 = !{!"llvm.loop.vectorize.enable", i1 true}
!2 = !{!"llvm.loop.vectorize.width", i32 4}

; Non-SIMD loop metadata (no vectorize.enable)
!3 = distinct !{!3, !4}
!4 = !{!"llvm.loop.unroll.count", i32 2}
