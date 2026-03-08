//===-- LowerCompUpdate.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// comp.update 降级实现
// 将迭代更新操作降级为 Affine 循环嵌套 + memref.store
//
//===----------------------------------------------------------------------===//


#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Dialects/Comp/include/Comp.h"
#include "Utils/BuilderUtil.h"
#include "Utils/LowerUtil.h"

namespace ezcompile {

/// 降级 Pattern：将 comp.update 转换为循环嵌套 + 存储
///
/// 实现思路：
/// 1. 根据 range 属性创建空间维度的循环嵌套
/// 2. 内联 update 的 region 到循环体内
/// 3. 降级 coord 和 sample 操作
/// 4. 将 yield 值存储到 memref[time, space...]
struct LowerUpdatePattern : mlir::OpConversionPattern<comp::UpdateOp> {
	using OpConversionPattern<comp::UpdateOp>::OpConversionPattern;

	mlir::LogicalResult matchAndRewrite(comp::UpdateOp op,
	                              OpAdaptor adaptor,
	                              mlir::ConversionPatternRewriter& rewriter) const override {
		mlir::Location loc = op.getLoc();

		mlir::memref::AllocOp alloc = getDefiningFieldOp(op.getField());
		if (!alloc) {
			return rewriter.notifyMatchFailure(op, "field is not backed by memref.alloc");
		}
		mlir::Value memref = alloc.getResult();
		auto memrefTy = dyn_cast<mlir::MemRefType>(memref.getType());
		if (!memrefTy) {
			return rewriter.notifyMatchFailure(op, "field alloc is not a memref");
		}

		// 创建空间维度循环
		mlir::SmallVector<mlir::Value, 8> argValues;
		for (mlir::Attribute r : op.getOver()) {
			auto range = dyn_cast<comp::RangeAttr>(r);
			if (!range) continue;

			comp::DimOp dimOp = lookupDimOp(op, range.getDim());

			auto lb = -range.getLower().getInt();
			auto ub = dimOp.getPoints() - range.getUpper().getInt();
			auto forOp = mlir::affine::AffineForOp::create(rewriter, loc, lb, ub, 1);
			mlir::Value iv = forOp.getInductionVar();
			argValues.emplace_back(iv);

			rewriter.setInsertionPointToStart(forOp.getBody());
		}

		mlir::Block& srcBlock = op.getRegion().front();
		mlir::Block* dstBlock = rewriter.getInsertionBlock();
		mlir::Operation* insertPt = dstBlock->getTerminator();
		mlir::Operation* marker = insertPt->getPrevNode();

		rewriter.inlineBlockBefore(&srcBlock, insertPt, argValues);

		// 收集 coord、sample 和 yield
		mlir::SmallVector<comp::CoordOp, 8> coords;
		mlir::SmallVector<comp::SampleOp, 8> samples;
		comp::YieldOp yieldOp;
		mlir::Operation *firstNonCoord = nullptr;

		mlir::Operation* cur = marker ? marker->getNextNode() : &dstBlock->front();
		while (cur && cur != insertPt) {
			if (auto c = dyn_cast<comp::CoordOp>(cur)) {
				coords.push_back(c);
				firstNonCoord = cur;
			}
			if (auto y = dyn_cast<comp::YieldOp>(cur)) yieldOp = y;
			if (auto s = dyn_cast<comp::SampleOp>(cur)) {
				samples.emplace_back(s);
			}
			cur = cur->getNextNode();
		}
		if (firstNonCoord == nullptr) {
			return mlir::emitError(loc, "doesn't have CoordOp");
		}
		firstNonCoord = firstNonCoord->getNextNode();

		if (!yieldOp) {
			return rewriter.notifyMatchFailure(op, "rhs region has no comp.yield after inlining");
		}
		if (yieldOp.getNumOperands() != 1) {
			return rewriter.notifyMatchFailure(op, "rhs comp.yield must have exactly 1 operand");
		}

		// 降级 coord
		mlir::Operation *hoistBefore = firstNonCoord ? firstNonCoord : yieldOp.getOperation();
		rewriter.setInsertionPoint(hoistBefore);

		for (comp::CoordOp c : coords) {
			mlir::Value iv = c.getIv();
			mlir::Value coordVal = lowerCoord(rewriter, c.getLoc(), op, c.getDimAttr(), iv);
			if (!coordVal) return mlir::failure();
			c.replaceAllUsesWith(coordVal);
		}
		for (comp::CoordOp c : coords) {
			rewriter.eraseOp(c);
		}

		// 降级 sample
		mlir::SmallVector<mlir::Value, 8> indices;
		auto time_var = modIndex(rewriter, loc, op.getAtTime(), 2);
		indices.emplace_back(time_var);
		for (auto arg : argValues) {
			indices.push_back(arg);
		}

		for (comp::SampleOp s : samples) {
			mlir::Value sampleVal = lowerSample(rewriter, loc, s, indices);
			if (!sampleVal) return mlir::failure();
			s.replaceAllUsesWith(sampleVal);
		}
		for (comp::SampleOp s : samples) {
			rewriter.eraseOp(s);
		}

		// 存储 yield 值 - 写入下一个时间步 + 1) % 2
		rewriter.setInsertionPoint(yieldOp);

		mlir::Value yieldedF64 = castToF64(rewriter, yieldOp.getLoc(), yieldOp.getOperand(0));

		// 修改时间索引为 (atTime + 1) % 2
		mlir::Value one = constIndex(rewriter, loc, 1);
		mlir::Value atTimePlusOne = mlir::arith::AddIOp::create(rewriter, loc, op.getAtTime(), one);
		mlir::Value storeTimeVar = modIndex(rewriter, loc, atTimePlusOne, 2);
		indices[0] = storeTimeVar;

		mlir::memref::StoreOp::create(rewriter, yieldOp.getLoc(), yieldedF64, memref, indices);
		rewriter.eraseOp(yieldOp);

		rewriter.eraseOp(op);
		return mlir::success();
	}
};

struct LowerCompUpdatePass : mlir::PassWrapper<LowerCompUpdatePass, mlir::OperationPass<mlir::ModuleOp>> {
	MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerCompUpdatePass)

	void getDependentDialects(mlir::DialectRegistry& registry) const override {
		registry.insert<mlir::affine::AffineDialect, mlir::memref::MemRefDialect, mlir::arith::ArithDialect>();
	}

	mlir::StringRef getArgument() const override { return "lower-comp-update"; }

	void runOnOperation() override {
		mlir::MLIRContext* context = &getContext();
		mlir::ModuleOp module = getOperation();

		mlir::ConversionTarget target(*context);

		target.addLegalDialect<mlir::affine::AffineDialect, mlir::memref::MemRefDialect, mlir::arith::ArithDialect>();
		target.addIllegalOp<comp::UpdateOp>();

		mlir::RewritePatternSet patterns(context);
		patterns.add<LowerUpdatePattern>(context);

		if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
			signalPassFailure();
		}
	}
};

void registerLowerCompUpdatePass() {
	mlir::PassRegistration<LowerCompUpdatePass>();
}

std::unique_ptr<mlir::Pass> createLowerCompUpdatePass() {
	return std::make_unique<LowerCompUpdatePass>();
}

} // namespace ezcompile
