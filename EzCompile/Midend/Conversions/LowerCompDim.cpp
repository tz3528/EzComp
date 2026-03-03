//===-- LowerCompDim.cpp --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// comp.dim 降级实现
// 将维度声明操作降级为常量定义
//
//===----------------------------------------------------------------------===//


#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Dialects/Comp/include/Comp.h"

namespace ezcompile {

/// 降级 Pattern：删除 comp.dim 操作
///
/// 实现思路：
/// comp.dim 的信息（lower/upper/points）已在前置 Pass 中被其他操作引用，
/// 此处只需删除该操作即可。
struct LowerDimPattern : mlir::OpConversionPattern<comp::DimOp> {
	using OpConversionPattern<comp::DimOp>::OpConversionPattern;

	mlir::LogicalResult matchAndRewrite(comp::DimOp op,
	                              OpAdaptor adaptor,
	                              mlir::ConversionPatternRewriter& rewriter) const override {
		rewriter.eraseOp(op);
		return mlir::success();
	}
};

struct LowerCompDimPass : mlir::PassWrapper<LowerCompDimPass, mlir::OperationPass<mlir::ModuleOp>> {
	MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerCompDimPass)

	void getDependentDialects(mlir::DialectRegistry& registry) const override {
		registry.insert<
				mlir::affine::AffineDialect, mlir::memref::MemRefDialect,
				mlir::arith::ArithDialect, mlir::func::FuncDialect, mlir::LLVM::LLVMDialect
		>();
	}

	mlir::StringRef getArgument() const override { return "lower-comp-dim"; }

	void runOnOperation() override {
		mlir::MLIRContext* context = &getContext();
		mlir::ModuleOp module = getOperation();

		mlir::ConversionTarget target(*context);

		target.addLegalDialect<
				mlir::affine::AffineDialect, mlir::memref::MemRefDialect,
				mlir::arith::ArithDialect, mlir::func::FuncDialect, mlir::LLVM::LLVMDialect>();
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

} // namespace ezcompile