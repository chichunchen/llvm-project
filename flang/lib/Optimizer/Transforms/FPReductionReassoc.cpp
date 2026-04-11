//===- FPReductionReassoc.cpp - Mark FP reductions with reassoc -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass identifies scalar floating-point reduction patterns inside
// fir.do_loop and sets the reassoc fast-math flag on the accumulation
// arith.addf (and the feeding arith.mulf when present). This enables
// LLVM's loop vectorizer to use multiple independent partial-sum
// accumulators.
//
// The Fortran standard does not mandate a specific accumulation order
// for such reductions, so reassociation is semantically valid.
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace fir {
#define GEN_PASS_DEF_FPREDUCTIONREASSOC
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

using namespace mlir;

namespace {

/// Return true if the value is guarded by a fir.no_reassoc op, which is
/// the user's explicit opt-out from reassociation.
static bool isGuardedByNoReassoc(Value val) {
  if (!val)
    return false;
  if (val.getDefiningOp<fir::NoReassocOp>())
    return true;
  return false;
}

struct FPReductionReassoc
    : public fir::impl::FPReductionReassocBase<FPReductionReassoc> {

  /// Mark reassoc on a matched addf and optionally its feeding mulf.
  static bool markReassoc(arith::AddFOp addf, Value other) {
    // Respect fir.no_reassoc — user explicitly opted out.
    if (isGuardedByNoReassoc(other))
      return false;

    // Optionally match a feeding arith.mulf (the Q + Z*X pattern).
    arith::MulFOp mulf = other.getDefiningOp<arith::MulFOp>();

    // Guard: only tag the mulf if it has a single use (the addf).
    if (mulf && !mulf->hasOneUse())
      mulf = nullptr;

    // Set reassoc flag on the addf.
    addf.setFastmath(addf.getFastmath() | arith::FastMathFlags::reassoc);

    // Set reassoc flag on the mulf if present.
    if (mulf)
      mulf.setFastmath(mulf.getFastmath() | arith::FastMathFlags::reassoc);

    return true;
  }

  /// Pattern 1: iter_arg-based reduction.
  /// The loop-carried accumulator is a float-typed iter_arg, and the yielded
  /// value is arith.addf with one operand being the iter_arg itself.
  bool matchIterArgReduction(fir::DoLoopOp loop) {
    auto resultOp = dyn_cast<fir::ResultOp>(loop.getBody()->getTerminator());
    if (!resultOp)
      return false;

    bool matched = false;
    for (auto [idx, regionIterArg] :
         llvm::enumerate(loop.getRegionIterArgs())) {
      if (!isa<FloatType>(regionIterArg.getType()))
        continue;

      Value yielded = resultOp.getOperand(idx);
      auto addf = yielded.getDefiningOp<arith::AddFOp>();
      if (!addf)
        continue;

      Value lhs = addf.getLhs();
      Value rhs = addf.getRhs();
      Value other;
      if (lhs == regionIterArg)
        other = rhs;
      else if (rhs == regionIterArg)
        other = lhs;
      else
        continue;

      matched |= markReassoc(addf, other);
    }
    return matched;
  }

  /// Pattern 2: memory-based reduction.
  /// A fir.load from a ref defined outside the loop feeds into arith.addf,
  /// whose result is stored back via fir.store to the same ref.
  bool matchMemoryReduction(fir::DoLoopOp loop) {
    Block *body = loop.getBody();
    Region &loopRegion = loop.getRegion();
    bool matched = false;

    for (auto &op : *body) {
      auto storeOp = dyn_cast<fir::StoreOp>(op);
      if (!storeOp)
        continue;

      Value storedVal = storeOp.getValue();
      Value storeAddr = storeOp.getMemref();

      // The store address must be defined outside the loop.
      if (loopRegion.isAncestor(storeAddr.getParentRegion()))
        continue;

      // Stored value must be floating-point.
      if (!isa<FloatType>(storedVal.getType()))
        continue;

      auto addf = storedVal.getDefiningOp<arith::AddFOp>();
      if (!addf)
        continue;

      // One operand of the addf must be a fir.load from the same address.
      Value lhs = addf.getLhs();
      Value rhs = addf.getRhs();
      Value other;

      auto isLoadFromSameAddr = [&](Value v) -> bool {
        auto loadOp = v.getDefiningOp<fir::LoadOp>();
        return loadOp && loadOp.getMemref() == storeAddr;
      };

      if (isLoadFromSameAddr(lhs))
        other = rhs;
      else if (isLoadFromSameAddr(rhs))
        other = lhs;
      else
        continue;

      matched |= markReassoc(addf, other);
    }
    return matched;
  }

  void runOnOperation() override {
    auto func = getOperation();
    bool changed = false;

    func.walk([&](fir::DoLoopOp loop) {
      changed |= matchIterArgReduction(loop);
      changed |= matchMemoryReduction(loop);
    });

    if (!changed)
      markAllAnalysesPreserved();
  }
};

} // namespace
