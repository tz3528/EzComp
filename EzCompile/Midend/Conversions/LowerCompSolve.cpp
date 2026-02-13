//===-- LowerCompSolve.cpp -------------------------------------*- C++ -*-===//
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

struct LowerSolvePattern : mlir::OpConversionPattern<comp::SolveOp> {
	using OpConversionPattern<comp::SolveOp>::OpConversionPattern;

	mlir::LogicalResult matchAndRewrite(comp::SolveOp op,
								  OpAdaptor adaptor,
								  mlir::ConversionPatternRewriter& rewriter) const override {
		mlir::Block *parentBlock = op->getBlock();
		mlir::Block::iterator insertIt(op); // 初始插入点：solve 之前

		auto moveSingleBlockRegionOps = [&](mlir::Region &region) -> mlir::LogicalResult {
			if (region.empty())
				return mlir::success();

			if (!llvm::hasSingleElement(region)) {
				return op.emitError("expected a single-block region for solve subregion");
			}

			mlir::Block &srcBlock = region.front();

			// 逐个 operation 搬运，保持原有顺序：
			while (!srcBlock.empty()) {
				mlir::Operation *toMove = &srcBlock.front();
				toMove->moveBefore(parentBlock, insertIt);
				insertIt = std::next(mlir::Block::iterator(toMove));
			}
			return mlir::success();
		};

		if (mlir::failed(moveSingleBlockRegionOps(op.getInit()))) {
			return mlir::failure();
		}
		if (mlir::failed(moveSingleBlockRegionOps(op.getBoundary()))) {
			return mlir::failure();
		}
		if (mlir::failed(moveSingleBlockRegionOps(op.getStep()))) {
			return mlir::failure();
		}

		rewriter.eraseOp(op);
		return mlir::success();
	}
};

struct LowerCompSolvePass : mlir::PassWrapper<LowerCompSolvePass, mlir::OperationPass<mlir::ModuleOp>> {
	MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerCompSolvePass)

	void getDependentDialects(mlir::DialectRegistry& registry) const override {
		registry.insert<mlir::affine::AffineDialect, mlir::memref::MemRefDialect, mlir::arith::ArithDialect>();
	}

	mlir::StringRef getArgument() const override { return "lower-comp-solve"; }

	void runOnOperation() override {
		mlir::MLIRContext* context = &getContext();
		mlir::ModuleOp module = getOperation();

		mlir::ConversionTarget target(*context);

		target.addLegalDialect<mlir::affine::AffineDialect, mlir::memref::MemRefDialect, mlir::arith::ArithDialect>();
		target.addIllegalOp<comp::SolveOp>();

		mlir::RewritePatternSet patterns(context);
		patterns.add<LowerSolvePattern>(context);

		if (mlir::failed(applyPartialConversion(module, target, std::move(patterns)))) {
			signalPassFailure();
		}
	}
};

void registerLowerCompSolvePass() {
	mlir::PassRegistration<LowerCompSolvePass>();
}

std::unique_ptr<mlir::Pass> createLowerCompSolvePass() {
	return std::make_unique<LowerCompSolvePass>();
}

}
