//===-- LowerCompCall.cpp --------------------------------------*- C++ -*-===//
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
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Dialects/Comp/include/Comp.h"
#include "Utils/LowerUtil.h"

namespace ezcompile {

struct LowerCallDeltaPattern : mlir::OpConversionPattern<comp::CallOp> {
	using OpConversionPattern<comp::CallOp>::OpConversionPattern;

	LowerCallDeltaPattern(mlir::MLIRContext *context) : OpConversionPattern<comp::CallOp>(context, 1) {}

	mlir::LogicalResult matchAndRewrite(comp::CallOp op,
										OpAdaptor adaptor,
										mlir::ConversionPatternRewriter &rewriter) const override {
		mlir::Location loc = op.getLoc();

		// 1. 只匹配 delta
		if (op.getCallee().str() != "delta") {
			return mlir::emitError(loc, "not delta");
		}

		// 2. 参数校验
		if (adaptor.getOperands().size() != 2) {
			return mlir::emitError(loc, "delta expects 2 operands");
		}

		mlir::Value var = adaptor.getOperands()[0];
		mlir::Value rank = adaptor.getOperands()[1];

		auto var_f = castToF64(rewriter, loc, var);
		auto rank_f = castToF64(rewriter, loc, rank);

		mlir::Value result = rewriter.create<mlir::math::PowFOp>(loc, var_f, rank_f);

		rewriter.replaceOp(op, result);
		return mlir::success();
	}
};

struct LowerCallUnknownPattern : mlir::OpConversionPattern<comp::CallOp> {
	using OpConversionPattern<comp::CallOp>::OpConversionPattern;

	LowerCallUnknownPattern(mlir::MLIRContext *context) : OpConversionPattern<comp::CallOp>(context, 0) {}

	mlir::LogicalResult matchAndRewrite(comp::CallOp op,
										OpAdaptor adaptor,
										mlir::ConversionPatternRewriter &rewriter) const override {
		llvm::StringRef callee = op.getCallee();
		if (callee.empty()) {
			return rewriter.notifyMatchFailure(op, "call has no callee name");
		}

		if (callee.str() == "diff") {
			// 这里的diff被认为必须有定义，且在ir的转换过程中就已被处理
			return mlir::success();
		}

		// 这里“总是匹配”，作为最后一个 pattern
		op.emitError() << "unsupported comp.call callee: " << callee;
		return mlir::failure();
	}
};

struct LowerCompCallPass : mlir::PassWrapper<LowerCompCallPass, mlir::OperationPass<mlir::ModuleOp>> {
	MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerCompCallPass)

	void getDependentDialects(mlir::DialectRegistry& registry) const override {
		registry.insert<mlir::math::MathDialect, mlir::arith::ArithDialect>();
	}

	mlir::StringRef getArgument() const override { return "lower-comp-call"; }

	void runOnOperation() override {
		mlir::MLIRContext* context = &getContext();
		mlir::ModuleOp module = getOperation();

		mlir::ConversionTarget target(*context);

		target.addLegalDialect<mlir::math::MathDialect, mlir::arith::ArithDialect>();
		target.addIllegalOp<comp::CallOp>();

		mlir::RewritePatternSet patterns(context);
		patterns.add<LowerCallDeltaPattern>(context);
		patterns.add<LowerCallUnknownPattern>(context);

		if (mlir::failed(applyPartialConversion(module, target, std::move(patterns)))) {
			signalPassFailure();
		}
	}
};

void registerLowerCompCallPass() {
	mlir::PassRegistration<LowerCompCallPass>();
}

std::unique_ptr<mlir::Pass> createLowerCompCallPass() {
	return std::make_unique<LowerCompCallPass>();
}

}
