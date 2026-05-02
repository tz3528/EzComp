//===-- LowerCompCall.cpp --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// comp.call 降级实现
// 将内置函数调用降级为算术运算
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

//===----------------------------------------------------------------------===//
// 三角函数降级 Pattern
//===----------------------------------------------------------------------===//

/// 单参数三角函数降级模板
/// 支持 sin, cos, tan, asin, acos, atan
template <typename MathOp>
struct LowerCallUnaryTrigPattern : mlir::OpConversionPattern<comp::CallOp> {
	using OpConversionPattern<comp::CallOp>::OpConversionPattern;

	llvm::StringRef funcName;

	LowerCallUnaryTrigPattern(mlir::MLIRContext *context, llvm::StringRef name)
		: OpConversionPattern<comp::CallOp>(context, 1), funcName(name) {}

	mlir::LogicalResult matchAndRewrite(comp::CallOp op,
										OpAdaptor adaptor,
										mlir::ConversionPatternRewriter &rewriter) const override {
		mlir::Location loc = op.getLoc();

		if (op.getCallee() != funcName) {
			return rewriter.notifyMatchFailure(op, "not " + funcName);
		}

		if (adaptor.getOperands().size() != 1) {
			return mlir::emitError(loc, funcName) << " expects 1 operand";
		}

		mlir::Value arg = adaptor.getOperands()[0];
		mlir::Value arg_f = castToF64(rewriter, loc, arg);

		mlir::Value result = rewriter.create<MathOp>(loc, arg_f);

		rewriter.replaceOp(op, result);
		return mlir::success();
	}
};

/// 双参数三角函数降级模板
/// 支持 atan2
template <typename MathOp>
struct LowerCallBinaryTrigPattern : mlir::OpConversionPattern<comp::CallOp> {
	using OpConversionPattern<comp::CallOp>::OpConversionPattern;

	llvm::StringRef funcName;

	LowerCallBinaryTrigPattern(mlir::MLIRContext *context, llvm::StringRef name)
		: OpConversionPattern<comp::CallOp>(context, 1), funcName(name) {}

	mlir::LogicalResult matchAndRewrite(comp::CallOp op,
										OpAdaptor adaptor,
										mlir::ConversionPatternRewriter &rewriter) const override {
		mlir::Location loc = op.getLoc();

		if (op.getCallee() != funcName) {
			return rewriter.notifyMatchFailure(op, "not " + funcName);
		}

		if (adaptor.getOperands().size() != 2) {
			return mlir::emitError(loc, funcName) << " expects 2 operands";
		}

		mlir::Value arg0 = adaptor.getOperands()[0];
		mlir::Value arg1 = adaptor.getOperands()[1];

		mlir::Value arg0_f = castToF64(rewriter, loc, arg0);
		mlir::Value arg1_f = castToF64(rewriter, loc, arg1);

		mlir::Value result = rewriter.create<MathOp>(loc, arg0_f, arg1_f);

		rewriter.replaceOp(op, result);
		return mlir::success();
	}
};

// 单参数三角函数 Pattern 别名
using LowerCallSinPattern = LowerCallUnaryTrigPattern<mlir::math::SinOp>;
using LowerCallCosPattern = LowerCallUnaryTrigPattern<mlir::math::CosOp>;
using LowerCallTanPattern = LowerCallUnaryTrigPattern<mlir::math::TanOp>;
using LowerCallAsinPattern = LowerCallUnaryTrigPattern<mlir::math::AsinOp>;
using LowerCallAcosPattern = LowerCallUnaryTrigPattern<mlir::math::AcosOp>;
using LowerCallAtanPattern = LowerCallUnaryTrigPattern<mlir::math::AtanOp>;

// 双参数三角函数 Pattern 别名
using LowerCallAtan2Pattern = LowerCallBinaryTrigPattern<mlir::math::Atan2Op>;

//===----------------------------------------------------------------------===//
// 指数与对数函数降级 Pattern
//===----------------------------------------------------------------------===//

// 单参数指数/对数函数 Pattern 别名
using LowerCallExpPattern = LowerCallUnaryTrigPattern<mlir::math::ExpOp>;
using LowerCallLogPattern = LowerCallUnaryTrigPattern<mlir::math::LogOp>;

/// 降级 Pattern：处理未知的 comp.call
///
/// 实现思路：
/// 对 diff 调用不做处理（已在其他 Pass 中处理），
/// 对其他未知调用报错。
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
			// diff 已在 IR 生成阶段处理
			return mlir::success();
		}

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

		target.addLegalDialect<mlir::math::MathDialect, mlir::arith::ArithDialect, comp::CompDialect>();
		target.addIllegalOp<comp::CallOp>();

		mlir::RewritePatternSet patterns(context);
		// 三角函数降级 Pattern
		patterns.add<LowerCallSinPattern>(context, "sin");
		patterns.add<LowerCallCosPattern>(context, "cos");
		patterns.add<LowerCallTanPattern>(context, "tan");
		patterns.add<LowerCallAsinPattern>(context, "asin");
		patterns.add<LowerCallAcosPattern>(context, "acos");
		patterns.add<LowerCallAtanPattern>(context, "atan");
		patterns.add<LowerCallAtan2Pattern>(context, "atan2");
		// 指数与对数函数降级 Pattern
		patterns.add<LowerCallExpPattern>(context, "exp");
		patterns.add<LowerCallLogPattern>(context, "log");
		// 未知调用处理
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

} // namespace ezcompile