//===-- LowerCompApplyInit.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// comp.apply_init 降级实现
// 将初始化操作降级为 Affine 循环嵌套 + memref.store
//
//===----------------------------------------------------------------------===//


#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Dialects/Comp/include/Comp.h"
#include "Utils/BuilderUtil.h"
#include "Utils/LowerUtil.h"

namespace ezcompile {

/// 降级 Pattern：将 comp.apply_init 转换为循环嵌套 + 存储
///
/// 实现思路：
/// 1. 解析锚点信息，区分固定维度和未固定维度
/// 2. 对未固定维度创建 Affine 循环嵌套
/// 3. 内联 apply_init 的 region 到循环体内
/// 4. 降级 coord 操作为坐标计算
/// 5. 将 yield 值存储到 memref[time=0, ...]
struct LowerApplyInitPattern : mlir::OpConversionPattern<comp::ApplyInitOp> {
	using OpConversionPattern<comp::ApplyInitOp>::OpConversionPattern;

	mlir::LogicalResult matchAndRewrite(comp::ApplyInitOp op, OpAdaptor adaptor,
								  mlir::ConversionPatternRewriter &rewriter) const override {
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

		// 解析锚点：dimSym -> 固定索引
		mlir::DenseMap<mlir::Attribute, uint64_t> fixed;
		for (mlir::Attribute a : op.getAnchors()) {
			auto anc = dyn_cast<comp::AnchorAttr>(a);
			if (!anc) continue;
			fixed[anc.getDim()] = anc.getIndex();
		}

		// 收集维度顺序
		mlir::SmallVector<mlir::FlatSymbolRefAttr, 4> unfixedDims;
		mlir::SmallVector<mlir::FlatSymbolRefAttr, 4> dims;
		mlir::Operation* symTableOp = op->getParentOfType<comp::ProblemOp>();
		if (!symTableOp) {
			return rewriter.notifyMatchFailure(op, "cannot find symbol table for dim ordering");
		}
		symTableOp->walk([&](comp::DimOp dim) {
			mlir::FlatSymbolRefAttr sym = mlir::FlatSymbolRefAttr::get(dim.getSymNameAttr());
			if (!fixed.contains(sym)) unfixedDims.emplace_back(sym);
			dims.emplace_back(sym);
		});

		if (unfixedDims.size() + fixed.size() != memrefTy.getRank()) {
			op.emitError() << "space dims count (" << unfixedDims.size()
				<< ") does not match memref rank (" << (memrefTy.getRank() - 1) << ")";
			return mlir::failure();
		}

		// dim -> 索引值映射
		mlir::DenseMap<mlir::Attribute, mlir::Value> dimIndexVal;

		// 固定维度的索引
		for (auto& kv : fixed) {
			auto d = dyn_cast<mlir::FlatSymbolRefAttr>(kv.first);
			if (!d) continue;
			dimIndexVal[d] = constIndex(rewriter, loc, kv.second);
		}

		// 对未固定维度创建循环
		for (mlir::FlatSymbolRefAttr d : unfixedDims) {
			comp::DimOp dimOp = lookupDimOp(op, d);
			auto points = static_cast<int64_t>(dimOp.getPoints());

			auto forOp = mlir::affine::AffineForOp::create(rewriter, loc, /*lb=*/0, /*ub=*/points, /*step=*/1);
			mlir::Value iv = forOp.getInductionVar();
			dimIndexVal[d] = iv;

			rewriter.setInsertionPointToStart(forOp.getBody());
		}

		// 内联 region 到循环内
		mlir::Block& srcBlock = op.getRhs().front();
		mlir::Block* dstBlock = rewriter.getInsertionBlock();
		mlir::Operation* insertPt = dstBlock->getTerminator();

		// 替换 coord 的 BlockArgument
		mlir::SmallVector<mlir::Value, 8> argValues;
		for (mlir::Operation& inner : srcBlock.getOperations()) {
			auto coord = dyn_cast<comp::CoordOp>(inner);
			if (!coord) continue;
			mlir::Value iv = coord.getIv();
			if (auto barg = dyn_cast<mlir::BlockArgument>(iv)) {
				if (barg.getOwner() != &srcBlock) continue;
				auto it = dimIndexVal.find(coord.getDimAttr());
				argValues.emplace_back(it->second);
			}
		}

		mlir::Operation* marker = insertPt->getPrevNode();

		rewriter.inlineBlockBefore(&srcBlock, insertPt, argValues);

		// 收集 coord 和 yield
		mlir::SmallVector<comp::CoordOp, 8> coordsToLower;
		comp::YieldOp yieldOp;
		mlir::Operation *firstNonCoord = nullptr;

		mlir::Operation* cur = marker ? marker->getNextNode() : &dstBlock->front();
		while (cur && cur != insertPt) {
			if (auto c = dyn_cast<comp::CoordOp>(cur)) {
				coordsToLower.push_back(c);
				firstNonCoord = cur;
			}
			if (auto y = dyn_cast<comp::YieldOp>(cur)) yieldOp = y;
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

		// 降级 coord
		mlir::Operation *hoistBefore = firstNonCoord ? firstNonCoord : yieldOp.getOperation();
		rewriter.setInsertionPoint(hoistBefore);

		for (comp::CoordOp c : coordsToLower) {
			mlir::Value iv = c.getIv();
			mlir::Value coordVal = lowerCoord(rewriter, c.getLoc(), op, c.getDimAttr(), iv);
			if (!coordVal) return mlir::failure();
			c.replaceAllUsesWith(coordVal);
		}
		for (comp::CoordOp c : coordsToLower) {
			rewriter.eraseOp(c);
		}

		// 存储 yield 值到 memref
		rewriter.setInsertionPoint(yieldOp);

		mlir::Value yieldedF64 = castToF64(rewriter, yieldOp.getLoc(), yieldOp.getOperand(0));

		mlir::SmallVector<mlir::Value, 8> indices;
		indices.reserve(memrefTy.getRank());
		for (auto d : dims) {
			indices.push_back(dimIndexVal.lookup(d));
		}

		mlir::memref::StoreOp::create(rewriter, yieldOp.getLoc(), yieldedF64, memref, indices);
		rewriter.eraseOp(yieldOp);

		rewriter.eraseOp(op);
		return mlir::success();
	}
};

struct LowerCompApplyInitPass : mlir::PassWrapper<LowerCompApplyInitPass, mlir::OperationPass<mlir::ModuleOp>> {
	MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerCompApplyInitPass)

	void getDependentDialects(mlir::DialectRegistry& registry) const override {
		registry.insert<mlir::arith::ArithDialect, mlir::memref::MemRefDialect, comp::CompDialect>();
	}

	mlir::StringRef getArgument() const final { return "lower-comp-apply-init"; }

	mlir::StringRef getDescription() const final {
		return "Lower comp.apply_init into affine.for loops + memref.store";
	}

	void runOnOperation() override {
		mlir::MLIRContext* context = &getContext();
		mlir::ModuleOp module = getOperation();

		mlir::ConversionTarget target(*context);

		target.addLegalDialect<mlir::affine::AffineDialect, mlir::memref::MemRefDialect, mlir::arith::ArithDialect>();
		target.addIllegalOp<comp::ApplyInitOp>();

		mlir::RewritePatternSet patterns(context);
		patterns.add<LowerApplyInitPattern>(context);

		if (mlir::failed(applyPartialConversion(module, target, std::move(patterns)))) {
			signalPassFailure();
		}
	}
};

void registerLowerCompApplyInitPass() {
	mlir::PassRegistration<LowerCompApplyInitPass>();
}

std::unique_ptr<mlir::Pass> createLowerCompApplyInitPass() {
	return std::make_unique<LowerCompApplyInitPass>();
}

} // namespace ezcompile