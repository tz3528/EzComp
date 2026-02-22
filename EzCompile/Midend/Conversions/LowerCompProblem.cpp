//===-- LowerCompProblem.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// comp.problem 降级实现
// 将顶层 comp.problem 操作降级为 func.main 函数
//
//===----------------------------------------------------------------------===//


#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Dialects/Comp/include/Comp.h"

namespace ezcompile {

/// 降级 Pattern：将 comp.problem 转换为 func.main
///
/// 实现思路：
/// 1. 创建空的 main 函数（无参数、无返回值）
/// 2. 将 problem body 内的所有操作移动到 main 函数中
/// 3. 删除原 problem 操作
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

		// 创建 main 函数
		rewriter.setInsertionPoint(op);
		auto funcType = rewriter.getFunctionType(/*inputs=*/{}, /*results=*/{});
		auto mainFunc = rewriter.create<mlir::func::FuncOp>(op.getLoc(), "main", funcType);

		mlir::Block *entry = mainFunc.addEntryBlock();
		rewriter.setInsertionPointToEnd(entry);
		auto ret = rewriter.create<mlir::func::ReturnOp>(op.getLoc());

		// 将 problem 内部操作移到 main 函数
		llvm::SmallVector<mlir::Operation*, 16> opsToMove;
		opsToMove.reserve(problemBlock.getOperations().size());
		for (mlir::Operation &inner : problemBlock.getOperations())
			opsToMove.push_back(&inner);

		for (mlir::Operation *inner : opsToMove) {
			// 跳过终结符（如 yield）
			if (inner->hasTrait<mlir::OpTrait::IsTerminator>())
				continue;

			rewriter.moveOpBefore(inner, ret);
		}

		rewriter.eraseOp(op);
		return mlir::success();
	}
};

struct LowerCompProblemPass : mlir::PassWrapper<LowerCompProblemPass, mlir::OperationPass<mlir::ModuleOp>> {
	MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerCompProblemPass)

	void getDependentDialects(mlir::DialectRegistry& registry) const override {
		registry.insert<
			mlir::affine::AffineDialect, mlir::memref::MemRefDialect,
			mlir::arith::ArithDialect, mlir::func::FuncDialect
		>();
	}

	mlir::StringRef getArgument() const override { return "lower-comp-problem"; }

	void runOnOperation() override {
		mlir::MLIRContext* context = &getContext();
		mlir::ModuleOp module = getOperation();

		mlir::ConversionTarget target(*context);

		target.addLegalDialect<
			mlir::affine::AffineDialect, mlir::memref::MemRefDialect,
			mlir::arith::ArithDialect, mlir::func::FuncDialect>();
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

} // namespace ezcompile