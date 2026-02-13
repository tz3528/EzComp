//===-- LowerCompUpdate.cpp ------------------------------------*- C++ -*-===//
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
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Dialects/Comp/include/Comp.h"
#include "Utils/BuilderUtil.h"
#include "Utils/LowerUtil.h"

namespace ezcompile {

struct LowerUpdatePattern : mlir::OpConversionPattern<comp::UpdateOp> {
	using OpConversionPattern<comp::UpdateOp>::OpConversionPattern;

	mlir::LogicalResult matchAndRewrite(comp::UpdateOp op,
	                              OpAdaptor adaptor,
	                              mlir::ConversionPatternRewriter& rewriter) const override {
		mlir::Location loc = op.getLoc();

		// 1. 获取update的基本信息(field、memref、at_time)
		auto field = op.getField();
		mlir::memref::AllocOp alloc = getDefiningFieldOp(op.getField());
		if (!alloc) {
			return rewriter.notifyMatchFailure(op, "field is not backed by memref.alloc");
		}
		mlir::Value memref = alloc.getResult();
		auto memrefTy = dyn_cast<mlir::MemRefType>(memref.getType());
		if (!memrefTy) {
			return rewriter.notifyMatchFailure(op, "field alloc is not a memref");
		}

		// 2. 降级update并生成循环
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
			return mlir::emitError(loc, "don`t have CoordOp");
		}
		firstNonCoord = firstNonCoord->getNextNode();

		if (!yieldOp) {
			return rewriter.notifyMatchFailure(op, "rhs region has no comp.yield after inlining");
		}
		if (yieldOp.getNumOperands() != 1) {
			return rewriter.notifyMatchFailure(op, "rhs comp.yield must have exactly 1 operand");
		}

		// 将插入点放在第一个非Coord操作的位置
		mlir::Operation *hoistBefore = firstNonCoord ? firstNonCoord : yieldOp.getOperation();
		rewriter.setInsertionPoint(hoistBefore);

		// 降级Coord操作
		for (comp::CoordOp c : coords) {
			mlir::Value iv = c.getIv(); // 已通过内联 argValues 替换
			mlir::Value coordVal = lowerCoord(rewriter, c.getLoc(), op, c.getDimAttr(), iv);
			if (!coordVal) return mlir::failure();
			c.replaceAllUsesWith(coordVal);
		}
		for (comp::CoordOp c : coords) {
			rewriter.eraseOp(c);
		}

		// 降级Sample操作
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

		// 将插入点恢复到要创建 store 的位置（通常在 yield 位置）
		rewriter.setInsertionPoint(yieldOp);

		// 将 yield 值存储到 memref[time=0, space...]，然后删除 yield
		mlir::Value yieldedF64 = castToF64(rewriter, yieldOp.getLoc(), yieldOp.getOperand(0));

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

	void runOnOperation() override {
		mlir::MLIRContext* context = &getContext();
		mlir::ModuleOp module = getOperation();

		mlir::ConversionTarget target(*context);

		// 标记 Affine, Arith 为合法
		target.addLegalDialect<mlir::affine::AffineDialect, mlir::memref::MemRefDialect, mlir::arith::ArithDialect>();
		// 标记 comp.for_time 为非法，强制框架对其进行转换
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

}