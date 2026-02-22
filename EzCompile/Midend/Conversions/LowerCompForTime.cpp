//===-- LowerCompForTime.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// comp.for_time 降级实现
// 将时间循环操作降级为 Affine 循环
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
#include "Utils/LowerUtil.h"

namespace ezcompile {

/// 降级 Pattern：将 comp.for_time 转换为 affine.for
///
/// 实现思路：
/// 1. 从操作数获取 lb、ub、step
/// 2. 创建 affine.for，将原循环体内联进去
/// 3. 降级循环体内的 coord 操作
struct LowerForTimePattern : mlir::OpConversionPattern<comp::ForTimeOp> {
	using OpConversionPattern<comp::ForTimeOp>::OpConversionPattern;

	mlir::LogicalResult matchAndRewrite(comp::ForTimeOp op,
	                              OpAdaptor adaptor,
	                              mlir::ConversionPatternRewriter& rewriter) const override {
		// 获取步长（必须是常量）
		mlir::APInt stepInt;
		if (!matchPattern(adaptor.getStep(), mlir::m_ConstantInt(&stepInt))) {
			return rewriter.notifyMatchFailure(op, "step operand is not a constant integer");
		}
		int64_t step = stepInt.getSExtValue();

		mlir::Value lb = adaptor.getLb();
		mlir::Value ub = adaptor.getUb();

		if (!lb.getType().isIndex() || !ub.getType().isIndex()) {
			return rewriter.notifyMatchFailure(op, "bounds must be of index type");
		}

		mlir::Location loc = op.getLoc();
		auto affineFor = mlir::affine::AffineForOp::create(rewriter,
			loc,
			mlir::ValueRange{lb}, rewriter.getDimIdentityMap(),
			mlir::ValueRange{ub}, rewriter.getDimIdentityMap(),
			step
		);

		// 内联循环体
		mlir::Block* oldBody =  &op.getBody().front();
		mlir::Block* newBody = affineFor.getBody();

		mlir::SmallVector<mlir::Value, 1> newArgs;
		newArgs.push_back(newBody->getArgument(0));

		mlir::Operation *newYield = newBody->getTerminator();
		rewriter.inlineBlockBefore(oldBody, newBody, newYield->getIterator(), newArgs);

		// 降级 coord
		mlir::Operation* cur = &newBody->front();
		rewriter.setInsertionPointAfter(cur);
		mlir::Operation* end = &newBody->back();
		mlir::SmallVector<comp::CoordOp, 8> coordsToLower;
		while (cur && cur != end) {
			if (auto c = dyn_cast<comp::CoordOp>(cur)) {
				coordsToLower.emplace_back(c);
			}
			cur = cur->getNextNode();
		}

		for (comp::CoordOp c : coordsToLower) {
			mlir::Value iv = c.getIv();
			mlir::Value coordVal = lowerCoord(rewriter, c.getLoc(), op, c.getDimAttr(), iv);
			if (!coordVal) return mlir::failure();
			c.replaceAllUsesWith(coordVal);
		}
		for (comp::CoordOp c : coordsToLower) {
			rewriter.eraseOp(c);
		}

		// 删除内联进来的旧终结符
		mlir::Operation *inlinedOldTerminator = newYield->getPrevNode();
		if (inlinedOldTerminator && inlinedOldTerminator->hasTrait<mlir::OpTrait::IsTerminator>()) {
			rewriter.eraseOp(inlinedOldTerminator);
		}

		rewriter.replaceOp(op, affineFor.getResults());
		return mlir::success();
	}
};

struct LowerCompForTimePass : mlir::PassWrapper<LowerCompForTimePass, mlir::OperationPass<mlir::ModuleOp>> {
	MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerCompForTimePass)

	void getDependentDialects(mlir::DialectRegistry& registry) const override {
		registry.insert<mlir::affine::AffineDialect, mlir::arith::ArithDialect>();
	}

	mlir::StringRef getArgument() const override { return "lower-comp-for_time"; }

	void runOnOperation() override {
		mlir::MLIRContext* context = &getContext();
		mlir::ModuleOp module = getOperation();

		mlir::ConversionTarget target(*context);

		target.addLegalDialect<mlir::affine::AffineDialect, mlir::arith::ArithDialect>();
		target.addIllegalOp<comp::ForTimeOp>();

		mlir::RewritePatternSet patterns(context);
		patterns.add<LowerForTimePattern>(context);

		if (mlir::failed(applyPartialConversion(module, target, std::move(patterns)))) {
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

} // namespace ezcompile