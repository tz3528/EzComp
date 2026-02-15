//===-- LowerCompDim.cpp ----------------------------------------*- C++ -*-===//
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


#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Dialects/Comp/include/Comp.h"

namespace ezcompile {

struct LowerDimPattern : mlir::OpConversionPattern<comp::DimOp> {
	using OpConversionPattern<comp::DimOp>::OpConversionPattern;

	mlir::LogicalResult matchAndRewrite(comp::DimOp op,
	                              OpAdaptor adaptor,
	                              mlir::ConversionPatternRewriter& rewriter) const override {
		// 对于dim而言，已经为所有其它操作提供过了信息，不再需要生成其它信息
		rewriter.eraseOp(op);
		return mlir::success();
	}
};

struct LowerCompDimPass : mlir::PassWrapper<LowerCompDimPass, mlir::OperationPass<mlir::ModuleOp>> {
	MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerCompDimPass)

	void getDependentDialects(mlir::DialectRegistry& registry) const override {
		registry.insert<mlir::arith::ArithDialect>();
	}

	mlir::StringRef getArgument() const override { return "lower-comp-dim"; }

	void runOnOperation() override {
		mlir::MLIRContext* context = &getContext();
		mlir::ModuleOp module = getOperation();

		mlir::ConversionTarget target(*context);

		target.addLegalDialect<mlir::arith::ArithDialect>();
		target.addIllegalOp<comp::DimOp>();

		mlir::RewritePatternSet patterns(context);
		patterns.add<LowerDimPattern>(context);

		if (mlir::failed(applyPartialConversion(module, target, std::move(patterns)))) {
			signalPassFailure();
		}
	}
};

void registerLowerCompDimPass() {
	mlir::PassRegistration<LowerCompDimPass>();
}

std::unique_ptr<mlir::Pass> createLowerCompDimPass() {
	return std::make_unique<LowerCompDimPass>();
}

}
