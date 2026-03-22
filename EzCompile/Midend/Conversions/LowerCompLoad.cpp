//===-- LowerCompLoad.cpp --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// comp.load 降级实现
// 将字段加载操作降级为 memref.load
//
//===----------------------------------------------------------------------===//


#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Dialects/Comp/include/Comp.h"
#include "Utils/LowerUtil.h"

namespace ezcompile {

/// 降级 Pattern：将 comp.load 转换为 memref.load
///
/// 实现思路：
/// 1. 从 field 获取底层 memref.alloc
/// 2. 对时间维度（第一个索引）应用 mod 2（ping-pong 缓冲）
/// 3. 构建访问索引并生成 memref.load
struct LoadOpLowering : mlir::OpConversionPattern<comp::LoadOp> {
	using OpConversionPattern<comp::LoadOp>::OpConversionPattern;

	mlir::LogicalResult matchAndRewrite(comp::LoadOp op,
	                                    OpAdaptor adaptor,
	                                    mlir::ConversionPatternRewriter &rewriter) const override {
		mlir::Location loc = op.getLoc();

		// 1. 获取底层 memref.alloc
		mlir::memref::AllocOp alloc = getDefiningFieldOp(op.getField());
		if (!alloc) {
			return rewriter.notifyMatchFailure(op, "field is not backed by memref.alloc");
		}
		mlir::Value memref = alloc.getResult();

		// 2. 获取索引
		auto indices = adaptor.getIndices();
		if (indices.empty()) {
			return rewriter.notifyMatchFailure(op, "load must have at least one index");
		}

		// 3. 对时间维度（第一个索引）应用 mod 2
		mlir::Value c2 = mlir::arith::ConstantIndexOp::create(rewriter, loc, 2);
		mlir::Value timeIndex = mlir::arith::RemUIOp::create(rewriter, loc, indices[0], c2);

		// 4. 构建访问索引：[time%2, space_idx1, space_idx2, ...]
		llvm::SmallVector<mlir::Value, 4> accessIndices;
		accessIndices.push_back(timeIndex);
		for (size_t i = 1; i < indices.size(); ++i) {
			accessIndices.push_back(indices[i]);
		}

		// 5. 生成 memref.load
		auto load = mlir::memref::LoadOp::create(rewriter, loc, memref, accessIndices);

		rewriter.replaceOp(op, load.getResult());
		return mlir::success();
	}
};

struct LowerCompLoadPass : mlir::PassWrapper<LowerCompLoadPass, mlir::OperationPass<mlir::ModuleOp>> {
	MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerCompLoadPass)

	void getDependentDialects(mlir::DialectRegistry &registry) const override {
		registry.insert<mlir::affine::AffineDialect, mlir::memref::MemRefDialect, mlir::arith::ArithDialect>();
	}

	mlir::StringRef getArgument() const override { return "lower-comp-load"; }

	void runOnOperation() override {
		mlir::MLIRContext *context = &getContext();
		mlir::ModuleOp module = getOperation();

		mlir::ConversionTarget target(*context);
		target.addLegalDialect<mlir::affine::AffineDialect, mlir::memref::MemRefDialect, mlir::arith::ArithDialect>();
		target.addIllegalOp<comp::LoadOp>();

		mlir::RewritePatternSet patterns(context);
		patterns.add<LoadOpLowering>(context);

		if (mlir::failed(applyPartialConversion(module, target, std::move(patterns)))) {
			signalPassFailure();
		}
	}
};

void registerLowerCompLoadPass() {
	mlir::PassRegistration<LowerCompLoadPass>();
}

std::unique_ptr<mlir::Pass> createLowerCompLoadPass() {
	return std::make_unique<LowerCompLoadPass>();
}

} // namespace ezcompile
