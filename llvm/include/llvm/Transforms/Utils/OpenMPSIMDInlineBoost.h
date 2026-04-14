//===- OpenMPSIMDInlineBoost.h - Boost inlining in SIMD loops -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Boosts inline thresholds for function calls inside loops marked with
// llvm.loop.vectorize.enable metadata (i.e., OpenMP SIMD loops). This enables
// aggressive inlining of scalar function calls, allowing LoopVectorize to
// vectorize the inlined loop body without requiring vector function bodies.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_UTILS_OPENMPSIMDINLINEBOOST_H
#define LLVM_TRANSFORMS_UTILS_OPENMPSIMDINLINEBOOST_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class Function;

class OpenMPSIMDInlineBoost : public PassInfoMixin<OpenMPSIMDInlineBoost> {
public:
  LLVM_ABI PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_OPENMPSIMDINLINEBOOST_H
