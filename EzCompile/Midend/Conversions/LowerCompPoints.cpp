//===-- LowerCompPoints.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// comp.points 降级实现
// 将网格点数操作降级为 arith.constant
//
//===----------------------------------------------------------------------===//


#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Dialects/Comp/include/Comp.h"

namespace ezcompile {

/// 降级 Pattern：comp.points @dim -> arith.constant <points>
///
/// 实现思路：
/// 通过符号引用查找对应的 comp.dim，获取其 points 属性，
/// 替换为常量索引值。
struct PointsOpLowering : mlir::OpConversionPattern<comp::PointsOp> {
	PointsOpLowering(mlir::TypeConverter& tc, mlir::MLIRContext* ctx, mlir::SymbolTableCollection& st)
		: OpConversionPattern<comp::PointsOp>(tc, ctx), symbolTable(st) {
	}

	mlir::LogicalResult matchAndRewrite(comp::PointsOp op, OpAdaptor adaptor,
	                                    mlir::ConversionPatternRewriter& rewriter) const override {
		auto dimRefAttr = op.getDimAttr();
		if (!dimRefAttr) {
			return rewriter.notifyMatchFailure(op, "missing dim symbol ref attribute");
		}

		auto dimOp = symbolTable.lookupNearestSymbolFrom<comp::DimOp>(op, dimRefAttr);
		if (!dimOp) {
			return rewriter.notifyMatchFailure(op, "failed to resolve referenced comp.dim");
		}

		int64_t points = dimOp.getPoints();
		if (points < 0) {
			return rewriter.notifyMatchFailure(op, "invalid points (<0) on comp.dim");
		}

		rewriter.replaceOpWithNewOp<mlir::arith::ConstantIndexOp>(op, points);
		return mlir::success();
	}

private:
	mlir::SymbolTableCollection& symbolTable;
};

struct LowerCompPointsPass
	: mlir::PassWrapper<LowerCompPointsPass, mlir::OperationPass<mlir::ModuleOp>> {
	MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerCompPointsPass)

	void getDependentDialects(mlir::DialectRegistry& registry) const override {
		registry.insert<mlir::arith::ArithDialect, comp::CompDialect>();
	}

	mlir::StringRef getArgument() const final { return "lower-comp-points-to-arith"; }

	mlir::StringRef getDescription() const final {
		return "Lower comp.points to arith.constant (index)";
	}

	void runOnOperation() override {
		mlir::MLIRContext* ctx = &getContext();
		mlir::ModuleOp module = getOperation();

		mlir::TypeConverter typeConverter;
		typeConverter.addConversion([](mlir::Type t) { return t; });

		mlir::RewritePatternSet patterns(ctx);
		mlir::SymbolTableCollection symbolTable;

		patterns.add<PointsOpLowering>(typeConverter, ctx, symbolTable);

		mlir::ConversionTarget target(*ctx);
		target.addLegalDialect<mlir::arith::ArithDialect>();
		target.addIllegalOp<comp::PointsOp>();
		target.markUnknownOpDynamicallyLegal([](mlir::Operation*) { return true; });

		if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
			signalPassFailure();
		}
	}
};

void registerLowerCompPointsPass() {
	mlir::PassRegistration<LowerCompPointsPass>();
}

std::unique_ptr<mlir::Pass> createLowerCompPointsPass() {
	return std::make_unique<LowerCompPointsPass>();
}

} // namespace ezcompile