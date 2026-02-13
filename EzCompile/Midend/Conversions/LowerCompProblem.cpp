//===-- LowerCompProblem.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// 
//
//===----------------------------------------------------------------------===//


#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Dialects/Comp/include/Comp.h"

namespace ezcompile {

struct LowerProblemPattern : mlir::OpConversionPattern<comp::ProblemOp> {
	using OpConversionPattern<comp::ProblemOp>::OpConversionPattern;

	mlir::LogicalResult matchAndRewrite(comp::ProblemOp op,
								  OpAdaptor adaptor,
								  mlir::ConversionPatternRewriter& rewriter) const override {

		mlir::Region &body = op.getBody();

		if (!llvm::hasSingleElement(body)) {
			return rewriter.notifyMatchFailure(
				op, "comp.problem body has multiple blocks; cannot inline safely");
		}

		mlir::Block &problemBlock = body.front();

		// 把 comp.problem 的 block 里的 operations 内联到父操作
		rewriter.inlineBlockBefore(&problemBlock, op->getBlock(), op->getIterator());

		rewriter.eraseOp(op);
		return mlir::success();
	}
};

struct LowerCompProblemPass : mlir::PassWrapper<LowerCompProblemPass, mlir::OperationPass<mlir::ModuleOp>> {
	MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerCompProblemPass)

	void getDependentDialects(mlir::DialectRegistry& registry) const override {
		registry.insert<mlir::affine::AffineDialect, mlir::memref::MemRefDialect, mlir::arith::ArithDialect>();
	}

	mlir::StringRef getArgument() const override { return "lower-comp-problem"; }

	void runOnOperation() override {
		mlir::MLIRContext* context = &getContext();
		mlir::ModuleOp module = getOperation();

		mlir::ConversionTarget target(*context);

		target.addLegalDialect<mlir::affine::AffineDialect, mlir::memref::MemRefDialect, mlir::arith::ArithDialect>();
		target.addIllegalOp<comp::ProblemOp>();

		mlir::RewritePatternSet patterns(context);
		patterns.add<LowerProblemPattern>(context);

		if (mlir::failed(applyPartialConversion(module, target, std::move(patterns)))) {
			signalPassFailure();
		}
	}
};

void registerLowerCompProblemPass() {
	mlir::PassRegistration<LowerCompProblemPass>();
}

std::unique_ptr<mlir::Pass> createLowerCompProblemPass() {
	return std::make_unique<LowerCompProblemPass>();
}

}
