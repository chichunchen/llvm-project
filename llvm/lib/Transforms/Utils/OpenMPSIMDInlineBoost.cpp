//===- OpenMPSIMDInlineBoost.cpp - Boost inlining in SIMD loops
//------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass boosts inline thresholds for function calls inside loops marked
// with llvm.loop.vectorize.enable metadata (e.g., OpenMP SIMD loops). By
// adding the "function-inline-threshold" attribute to call sites, the inliner
// will aggressively inline these calls, enabling LoopVectorize to vectorize
// the loop body without requiring vector function bodies (simd clones).
//
// This implements the strategy used by Cray's CCE Fortran compiler, which
// aggressively inlines all function calls inside !$omp simd loops to enable
// vectorization.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/OpenMPSIMDInlineBoost.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "openmp-simd-inline-boost"

static cl::opt<unsigned> SIMDInlineThreshold(
    "openmp-simd-inline-threshold",
    cl::desc("Inline threshold for calls inside OpenMP SIMD loops"),
    cl::init(2000), cl::Hidden);

/// Returns true if the loop has llvm.loop.vectorize.enable set to true.
static bool hasVectorizeEnableMetadata(const Loop *L) {
  return getBooleanLoopAttribute(L, "llvm.loop.vectorize.enable");
}

/// Boost inline thresholds for all calls in the given loop and its subloops.
static bool boostCallsInLoop(Loop *L) {
  bool Changed = false;
  std::string ThresholdStr = std::to_string(SIMDInlineThreshold);

  for (BasicBlock *BB : L->blocks()) {
    for (Instruction &I : *BB) {
      auto *CB = dyn_cast<CallBase>(&I);
      if (!CB)
        continue;
      // Skip intrinsics and inline asm.
      if (CB->isInlineAsm() || isa<IntrinsicInst>(CB))
        continue;
      // Don't override if the call site already has a threshold set.
      if (CB->hasFnAttr("function-inline-threshold"))
        continue;
      CB->addFnAttr(Attribute::get(CB->getContext(),
                                   "function-inline-threshold", ThresholdStr));
      Changed = true;
    }
  }
  return Changed;
}

PreservedAnalyses OpenMPSIMDInlineBoost::run(Function &F,
                                             FunctionAnalysisManager &AM) {
  auto &LI = AM.getResult<LoopAnalysis>(F);

  bool Changed = false;
  for (Loop *L : LI) {
    // Check the outermost loop and all subloops for vectorize.enable metadata.
    // Boost calls in any loop (including nested) that is inside a SIMD loop.
    SmallVector<Loop *, 4> Worklist;
    Worklist.push_back(L);
    while (!Worklist.empty()) {
      Loop *CurLoop = Worklist.pop_back_val();
      if (hasVectorizeEnableMetadata(CurLoop)) {
        // Boost all calls in this loop and its subloops.
        Changed |= boostCallsInLoop(CurLoop);
      } else {
        // Check subloops — a nested loop might have the metadata.
        Worklist.append(CurLoop->getSubLoops().begin(),
                        CurLoop->getSubLoops().end());
      }
    }
  }

  if (!Changed)
    return PreservedAnalyses::all();

  // We only added attributes; we didn't modify the CFG or any analyses.
  return PreservedAnalyses::all();
}
