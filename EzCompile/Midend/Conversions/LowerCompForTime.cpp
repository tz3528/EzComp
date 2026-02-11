//===-- LowerCompForTime.cpp -----------------------------------*- C++ -*-===//
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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Dialects/Comp/include/Comp.h"

namespace ezcompile {

struct LowerForTimePattern : mlir::OpConversionPattern<comp::ForTimeOp> {
	using OpConversionPattern<comp::ForTimeOp>::OpConversionPattern;

	mlir::LogicalResult matchAndRewrite(comp::ForTimeOp op,
	                              OpAdaptor adaptor,
	                              mlir::ConversionPatternRewriter& rewriter) const override {
		// 1. 获取并验证 Step (步长)
		// affine.for 要求步长必须是 int64_t 常量。
		// 我们检查转换后的 step 操作数是否定义自一个常量。
		mlir::APInt stepInt;
		if (!matchPattern(adaptor.getStep(), mlir::m_ConstantInt(&stepInt))) {
			return rewriter.notifyMatchFailure(op, "step operand is not a constant integer");
		}
		int64_t step = stepInt.getSExtValue();

		// 2. 获取 Lower 和 Upper Bounds
		// 注意：必须使用 adaptor 获取转换后的操作数，而不是原始操作数
		mlir::Value lb = adaptor.getLb();
		mlir::Value ub = adaptor.getUb();

		if (!lb.getType().isIndex() || !ub.getType().isIndex()) {
			return rewriter.notifyMatchFailure(op, "bounds must be of index type");
		}

		// 3. 创建 affine.for
		mlir::Location loc = op.getLoc();
		auto affineFor = mlir::affine::AffineForOp::create(rewriter,
			loc,
			mlir::ValueRange{lb}, rewriter.getDimIdentityMap(),
			mlir::ValueRange{ub}, rewriter.getDimIdentityMap(),
			step
		);

		// 4. 迁移循环体
		mlir::Block* oldBody =  &op.getBody().front();
		mlir::Block* newBody = affineFor.getBody();

		// 准备参数重映射：将 oldBody 的 IV (%arg0) 映射到 newBody 的 IV
		mlir::SmallVector<mlir::Value, 1> newArgs;
		newArgs.push_back(newBody->getArgument(0));

		// newBody 创建时自带 affine.yield，把旧 body 内联到该 yield 之前
		mlir::Operation *newYield = newBody->getTerminator();
		rewriter.inlineBlockBefore(oldBody, newBody, newYield->getIterator(), newArgs);

		// 内联后，“旧 terminator”也会被搬进 newBody（位于 newYield 之前），删掉它
		//（通常是 comp.yield / 自定义 terminator）
		mlir::Operation *inlinedOldTerminator = newYield->getPrevNode();
		if (inlinedOldTerminator && inlinedOldTerminator->hasTrait<mlir::OpTrait::IsTerminator>()) {
			rewriter.eraseOp(inlinedOldTerminator);
		}

		// 5. 替换原 Op
		rewriter.replaceOp(op, affineFor.getResults());
		return mlir::success();
	}
};

struct LowerCompForTimePass : mlir::PassWrapper<LowerCompForTimePass, mlir::OperationPass<mlir::ModuleOp>> {
	MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerCompForTimePass)

	void getDependentDialects(mlir::DialectRegistry& registry) const override {
		registry.insert<mlir::affine::AffineDialect, mlir::arith::ArithDialect>();
	}

	void runOnOperation() override {
		mlir::MLIRContext* context = &getContext();
		mlir::ModuleOp module = getOperation();

		mlir::ConversionTarget target(*context);

		// 标记 Affine, Arith 为合法
		target.addLegalDialect<mlir::affine::AffineDialect, mlir::arith::ArithDialect>();
		// 标记 comp.for_time 为非法，强制框架对其进行转换
		target.addIllegalOp<comp::ForTimeOp>();

		mlir::RewritePatternSet patterns(context);
		patterns.add<LowerForTimePattern>(context);

		if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
			signalPassFailure();
		}
	}
};

void registerLowerCompForTimePass() {
	mlir::PassRegistration<LowerCompForTimePass>();
}

std::unique_ptr<mlir::Pass> createLowerCompForTimePass() {
	return std::make_unique<LowerCompForTimePass>();
}

}
