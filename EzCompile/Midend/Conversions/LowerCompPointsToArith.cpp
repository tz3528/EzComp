//===-- LowerCompPointsToArith.cpp -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// 将points操作降级为arith
//
//===----------------------------------------------------------------------===//


#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Dialects/Comp/include/Comp.h"

namespace ezcompile {

/// comp.points @t : index  => arith.constant <Nt> : index
struct PointsOpLowering : mlir::OpConversionPattern<comp::PointsOp> {
	PointsOpLowering(mlir::TypeConverter& tc, mlir::MLIRContext* ctx, mlir::SymbolTableCollection& st)
		: OpConversionPattern<comp::PointsOp>(tc, ctx), symbolTable(st) {
	}

	mlir::LogicalResult matchAndRewrite(comp::PointsOp op, OpAdaptor adaptor,
	                                    mlir::ConversionPatternRewriter& rewriter) const override {
		// 1) 取 comp.points 引用的维度符号
		auto dimRefAttr = op.getDimAttr();
		if (!dimRefAttr) {
			return rewriter.notifyMatchFailure(op, "missing dim symbol ref attribute");
		}

		// 2) 向上就近查对应的 comp.dim 符号
		auto dimOp = symbolTable.lookupNearestSymbolFrom<comp::DimOp>(op, dimRefAttr);
		if (!dimOp) {
			return rewriter.notifyMatchFailure(op, "failed to resolve referenced comp.dim");
		}

		// 3) 从 comp.dim 的 domain<... points=...> 里读 points
		int64_t points = dimOp.getPoints();
		if (points < 0) {
			return rewriter.notifyMatchFailure(op, "invalid points (<0) on comp.dim");
		}

		// 4) 替换为 arith 常量 index
		rewriter.replaceOpWithNewOp<mlir::arith::ConstantIndexOp>(op, points);
		return mlir::success();
	}

private:
	mlir::SymbolTableCollection& symbolTable;
};

struct LowerCompPointsToArithPass
	: mlir::PassWrapper<LowerCompPointsToArithPass, mlir::OperationPass<mlir::ModuleOp>> {
	MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerCompPointsToArithPass)

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

		// 你的项目如果已有统一的 TypeConverter，可以复用；这里只有 index，不需要改类型
		mlir::TypeConverter typeConverter;
		typeConverter.addConversion([](mlir::Type t) { return t; });

		mlir::RewritePatternSet patterns(ctx);
		mlir::SymbolTableCollection symbolTable;

		patterns.add<PointsOpLowering>(typeConverter, ctx, symbolTable);

		mlir::ConversionTarget target(*ctx);
		target.addLegalDialect<mlir::arith::ArithDialect>();
		// 其他方言/Op 的合法性按你现有 pipeline 来；这里最关键是把 comp.points 标为 illegal
		target.addIllegalOp<comp::PointsOp>();

		// 如果你的 pipeline 里还有其他 dialect 需要继续合法，按需加：
		// target.addLegalDialect<func::FuncDialect, scf::SCFDialect, memref::MemRefDialect, ...>();
		target.markUnknownOpDynamicallyLegal([](mlir::Operation*) { return true; });

		if (failed(applyPartialConversion(module, target, std::move(patterns))))
			signalPassFailure();
	}
};

void registerLowerCompPointsToArithPass() {
	mlir::PassRegistration<LowerCompPointsToArithPass>();
}

std::unique_ptr<mlir::Pass> createLowerCompPointsToArithPass() {
	return std::make_unique<LowerCompPointsToArithPass>();
}

}
