//===- FPReductionReassoc.cpp - Enable reassoc on FP reductions -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass identifies scalar floating-point reduction patterns inside
/// fir.do_loop operations and sets the 'reassoc' fast-math flag on the
/// accumulation (and optional multiply) operations.
///
/// The pattern targeted is the common Fortran dot-product idiom:
///
///   DO k = 1, n
///     Q = Q + Z(k) * X(k)   ! or Q = Q + expr
///   END DO
///
/// In FIR this appears as a fir.do_loop with a float-typed iterArg where the
/// loop-result value is produced by an arith.addf whose one operand is the
/// iterArg itself.
///
/// Adding 'reassoc' to the addf (and any mulf immediately feeding it) tells
/// LLVM's loop vectorizer that it may break the loop-carried dependency by
/// using multiple independent partial-sum accumulators, enabling
/// auto-vectorization of the reduction.
///
/// This transformation is semantically safe in the context of Fortran DO-loop
/// reductions: the Fortran standard does not mandate a specific evaluation
/// order for such scalar accumulations.
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"

namespace fir {
#define GEN_PASS_DEF_FPREDUCTIONREASSOC
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

#define DEBUG_TYPE "flang-fp-reduction-reassoc"

namespace {

/// Return the fir::ResultOp terminator of the loop body, or nullptr.
static fir::ResultOp getLoopResultOp(fir::DoLoopOp loop) {
  return mlir::dyn_cast<fir::ResultOp>(loop.getBody()->getTerminator());
}

/// Add 'reassoc' to the fastmath flags of \p op (which must implement
/// ArithFastMathInterface).
static void addReassoc(mlir::Operation *op) {
  auto fmi = mlir::dyn_cast<mlir::arith::ArithFastMathInterface>(*op);
  if (!fmi)
    return;
  mlir::arith::FastMathFlags cur = fmi.getFastMathFlagsAttr().getValue();
  if ((cur & mlir::arith::FastMathFlags::reassoc) ==
      mlir::arith::FastMathFlags::reassoc)
    return; // already set
  mlir::arith::FastMathFlags updated =
      cur | mlir::arith::FastMathFlags::reassoc;
  op->setAttr(fmi.getFastMathAttrName(),
              mlir::arith::FastMathFlagsAttr::get(op->getContext(), updated));
}

class FPReductionReassocPass
    : public fir::impl::FPReductionReassocBase<FPReductionReassocPass> {
public:
  void runOnOperation() override;
};

} // namespace

void FPReductionReassocPass::runOnOperation() {
  mlir::func::FuncOp func = getOperation();

  func.walk([&](fir::DoLoopOp loop) {
    // --- Pattern 1: fir.do_loop iterArg-based FP reduction ---
    // Covers explicit iterArgs (rare for user code, but common in generated
    // code like SimplifyIntrinsics output).
    if (fir::ResultOp resultOp = getLoopResultOp(loop)) {
      auto iterArgs = loop.getRegionIterArgs();
      auto resultOperands = resultOp.getOperands();
      if (iterArgs.size() == resultOperands.size()) {
        for (auto [iterArg, resultVal] : llvm::zip(iterArgs, resultOperands)) {
          if (!mlir::isa<mlir::FloatType>(iterArg.getType()))
            continue;
          auto addfOp = resultVal.getDefiningOp<mlir::arith::AddFOp>();
          if (!addfOp)
            continue;
          mlir::Value other;
          if (addfOp.getLhs() == iterArg)
            other = addfOp.getRhs();
          else if (addfOp.getRhs() == iterArg)
            other = addfOp.getLhs();
          else
            continue;
          addReassoc(addfOp.getOperation());
          if (auto mulfOp = other.getDefiningOp<mlir::arith::MulFOp>())
            addReassoc(mulfOp.getOperation());
        }
      }
    }

    // --- Pattern 2: memory-based FP reduction ---
    // Covers the common Fortran pattern:
    //   fir.load %accum_ref    (ref defined outside loop)
    //   arith.mulf ...
    //   arith.addf %loaded, %product   (or addf(%product, %loaded))
    //   fir.store %sum -> %accum_ref
    // This is what dummy-argument reductions like Q = Q + Z(k)*X(k) produce.
    mlir::Block *loopBody = loop.getBody();
    loopBody->walk([&](fir::StoreOp storeOp) {
      // The store's ref must be defined outside the loop body.
      mlir::Value ref = storeOp.getMemref();
      if (loopBody->findAncestorOpInBlock(*ref.getDefiningOp()) != nullptr)
        return;

      // The stored value must come from an arith.addf.
      auto addfOp = storeOp.getValue().getDefiningOp<mlir::arith::AddFOp>();
      if (!addfOp)
        return;

      // One operand of the addf must come from a fir.load of the same ref.
      auto isLoadFromRef = [&](mlir::Value v) -> bool {
        auto loadOp = v.getDefiningOp<fir::LoadOp>();
        return loadOp && loadOp.getMemref() == ref;
      };

      mlir::Value other;
      if (isLoadFromRef(addfOp.getLhs()))
        other = addfOp.getRhs();
      else if (isLoadFromRef(addfOp.getRhs()))
        other = addfOp.getLhs();
      else
        return;

      addReassoc(addfOp.getOperation());
      if (auto mulfOp = other.getDefiningOp<mlir::arith::MulFOp>())
        addReassoc(mulfOp.getOperation());
    });
  });
}
