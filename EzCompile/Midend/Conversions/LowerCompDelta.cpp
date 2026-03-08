//===-- LowerCompDelta.cpp -------------------------------------*- C++ -*-===//
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
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Dialects/Comp/include/Comp.h"
#include "Utils/BuilderUtil.h"

namespace ezcompile {

struct LowerDeltaPattern : mlir::OpConversionPattern<comp::DeltaOp> {
	using mlir::OpConversionPattern<comp::DeltaOp>::OpConversionPattern;

	mlir::LogicalResult matchAndRewrite(comp::DeltaOp op, OpAdaptor adaptor,
										mlir::ConversionPatternRewriter& rewriter) const override {
		mlir::Location loc = op.getLoc();
		auto f64Ty = rewriter.getF64Type();

		auto dimRefAttr = op.getDimAttr();
		if (!dimRefAttr) {
			return rewriter.notifyMatchFailure(op, "missing dim symbol ref attribute");
		}

		auto dimOp = mlir::SymbolTable::lookupNearestSymbolFrom<comp::DimOp>(op, dimRefAttr);
		if (!dimOp) {
			return rewriter.notifyMatchFailure(op, "failed to resolve referenced comp.dim");
		}

		auto rank_i = op.getRank();
		if(rank_i == 0){
			mlir::Value one = rewriter.create<mlir::arith::ConstantFloatOp>(
					loc, f64Ty, mlir::APFloat(1.0));
			rewriter.replaceOp(op, one);
			return mlir::success();
		}

		double delta = (dimOp.getUpper().convertToDouble() - dimOp.getLower().convertToDouble())
					   / (dimOp.getPoints() - 1);
		auto val = rewriter.create<mlir::arith::ConstantFloatOp>(
				loc, f64Ty, mlir::APFloat(delta));
		if(rank_i == 1){
			rewriter.replaceOp(op, val);
			return mlir::success();
		}

		mlir::Value result = val;
		for (int64_t i = 1; i < rank_i; ++i) {
			result = rewriter.create<mlir::arith::MulFOp>(loc, result, val);
		}

		rewriter.replaceOp(op, result);
		return mlir::success();
	}
};

struct LowerCompDeltaPass: mlir::PassWrapper<LowerCompDeltaPass, mlir::OperationPass<mlir::ModuleOp>> {
	MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerCompDeltaPass)

	void getDependentDialects(mlir::DialectRegistry& registry) const override {
		registry.insert<mlir::arith::ArithDialect, comp::CompDialect>();
	}

	mlir::StringRef getArgument() const final { return "lower-comp-delta-to-arith"; }

	mlir::StringRef getDescription() const final {
		return "Lower comp.points to arith.constant (index)";
	}

	void runOnOperation() override {
		mlir::MLIRContext* context = &getContext();
		mlir::ModuleOp module = getOperation();

		mlir::ConversionTarget target(*context);

		target.addLegalDialect<mlir::arith::ArithDialect>();
		target.addIllegalOp<comp::DeltaOp>();

		mlir::RewritePatternSet patterns(context);
		patterns.add<LowerDeltaPattern>(context);

		if (mlir::failed(applyPartialConversion(module, target, std::move(patterns)))) {
			signalPassFailure();
		}
	}
};

void registerLowerCompDeltaPass() {
	mlir::PassRegistration<LowerCompDeltaPass>();
}

std::unique_ptr<mlir::Pass> createLowerCompDeltaPass() {
	return std::make_unique<LowerCompDeltaPass>();
}

} // namespace ezcompile
