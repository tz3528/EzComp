//===-- LowerCompApplyInit.cpp ---------------------------------*- C++ -*-===//
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
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Dialects/Comp/include/Comp.h"

namespace ezcompile {

/// --------- 穿透 unrealized cast 拿到对应的 alloc ---------
static mlir::Value stripCasts(mlir::Value v) {
	while (auto cast = v.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
		if (cast.getNumOperands() == 0) break;
		v = cast.getOperand(0);
	}
	return v;
}

static mlir::memref::AllocOp getDefiningFieldOp(mlir::Value maybeFieldLike) {
	mlir::Value base = stripCasts(maybeFieldLike);
	return base.getDefiningOp<mlir::memref::AllocOp>();
}

// 附近通过符号引用查找 comp.dim
static comp::DimOp lookupDimOp(mlir::Operation* from, mlir::FlatSymbolRefAttr dimSym) {
	if (!dimSym) return {};
	mlir::Operation* sym = mlir::SymbolTable::lookupNearestSymbolFrom(from, dimSym.getAttr());
	return dyn_cast_or_null<comp::DimOp>(sym);
}

// 将任何数值/索引值转换为 f64（用于 yield 值和坐标计算）
static mlir::Value castToF64(mlir::OpBuilder& b, mlir::Location loc, mlir::Value v) {
	mlir::Type t = v.getType();
	mlir::Type f64 = b.getF64Type();

	if (t == f64) return v;

	if (auto ft = dyn_cast<mlir::FloatType>(t)) {
		if (ft.getWidth() < 64) {
			return mlir::arith::ExtFOp::create(b, loc, f64, v);
		}
		if (ft.getWidth() > 64) {
			return mlir::arith::TruncFOp::create(b, loc, f64, v);
		}
	}

	if (llvm::isa<mlir::IndexType>(t)) {
		auto i64 = b.getI64Type();
		mlir::Value asI64 = mlir::arith::IndexCastOp::create(b, loc, i64, v);
		return mlir::arith::SIToFPOp::create(b, loc, f64, asI64);
	}

	if (auto it = dyn_cast<mlir::IntegerType>(t)) {
		// 整型只存在有符号整型
		return mlir::arith::SIToFPOp::create(b, loc, f64, v);
	}

	// 应该不存在其它类型
	llvm_unreachable("Unsupported type for castToF64");
}

// coord操作降级，计算方式为 lb + (ub - lb) * (index / (points - 1))
static mlir::Value lowerCoord(mlir::OpBuilder& b, mlir::Location loc, mlir::Operation* anchorOp,
                              mlir::FlatSymbolRefAttr dimSym, mlir::Value ivIndex) {
	comp::DimOp dimOp = lookupDimOp(anchorOp, dimSym);
	if (!dimOp) {
		anchorOp->emitError() << "cannot resolve comp.dim for " << dimSym;
		return {};
	}

	double lower = dimOp.getLower().convertToDouble();
	double upper = dimOp.getUpper().convertToDouble();
	int64_t points = static_cast<int64_t>(dimOp.getPoints());

	mlir::Type f64 = b.getF64Type();
	mlir::Type i64 = b.getI64Type();

	mlir::Value cLower = mlir::arith::ConstantFloatOp::create(b, loc, cast<mlir::FloatType>(f64), mlir::APFloat(lower));
	mlir::Value cUpper = mlir::arith::ConstantFloatOp::create(b, loc, cast<mlir::FloatType>(f64), mlir::APFloat(upper));

	mlir::Value cPointsI64 = mlir::arith::ConstantIntOp::create(b, loc, i64, points);
	mlir::Value cOneI64 = mlir::arith::ConstantIntOp::create(b, loc, i64, 1);
	mlir::Value denomI64 = mlir::arith::SubIOp::create(b, loc, cPointsI64, cOneI64);

	// iv: index -> i64 -> f64
	mlir::Value ivI64 = mlir::arith::IndexCastOp::create(b, loc, i64, ivIndex);
	mlir::Value ivF64 = mlir::arith::SIToFPOp::create(b, loc, f64, ivI64);
	mlir::Value denomF64 = mlir::arith::SIToFPOp::create(b, loc, f64, denomI64);

	mlir::Value span = mlir::arith::SubFOp::create(b, loc, cUpper, cLower);
	mlir::Value ratio = mlir::arith::DivFOp::create(b, loc, ivF64, denomF64);
	mlir::Value offset = mlir::arith::MulFOp::create(b, loc, span, ratio);
	return mlir::arith::AddFOp::create(b, loc, cLower, offset);
}

struct LowerApplyInitPattern : mlir::OpConversionPattern<comp::ApplyInitOp> {
	using OpConversionPattern<comp::ApplyInitOp>::OpConversionPattern;

	mlir::LogicalResult matchAndRewrite(comp::ApplyInitOp op, OpAdaptor adaptor,
								  mlir::ConversionPatternRewriter &rewriter) const override {
		mlir::Location loc = op.getLoc();

		// 1) 获取 memref.alloc
		mlir::memref::AllocOp alloc = getDefiningFieldOp(op.getField());
		if (!alloc) {
			return rewriter.notifyMatchFailure(op, "field is not backed by memref.alloc");
		}
		mlir::Value memref = alloc.getResult();
		auto memrefTy = dyn_cast<mlir::MemRefType>(memref.getType());
		if (!memrefTy) {
			return rewriter.notifyMatchFailure(op, "field alloc is not a memref");
		}

		// 2) 将锚点解析为映射：dimSym -> 固定索引（作为 uint64）
		mlir::DenseMap<mlir::Attribute, uint64_t> fixed;
		for (mlir::Attribute a : op.getAnchors()) {
			auto anc = dyn_cast<comp::AnchorAttr>(a);
			if (!anc) continue;
			fixed[anc.getDim()] = anc.getIndex();
		}

		// 3) 确定未固定维度的顺序和所有dim的顺序
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

		// memref 的秩应为 1(time)+N(space)
		if (unfixedDims.size() + fixed.size() != memrefTy.getRank()) {
			op.emitError() << "space dims count (" << unfixedDims.size()
				<< ") does not match memref rank (" << (memrefTy.getRank() - 1) << ")";
			return mlir::failure();
		}

		// 4) 构建一个映射 dim -> 索引值（循环 iv 或常量）
		mlir::DenseMap<mlir::Attribute, mlir::Value> dimIndexVal;

		// 创建一个 lambda 来实例化固定索引常量
		auto constIndex = [&](uint64_t idx) -> mlir::Value {
			return mlir::arith::ConstantIndexOp::create(rewriter, loc, static_cast<int64_t>(idx));
		};

		// 所有固定维度的索引应当是固定值
		for (auto& kv : fixed) {
			auto d = dyn_cast<mlir::FlatSymbolRefAttr>(kv.first);
			if (!d) continue;
			dimIndexVal[d] = constIndex(kv.second);
		}

		// 对所有未固定的维度创建循环
		mlir::SmallVector<mlir::affine::AffineForOp, 4> loops;
		for (mlir::FlatSymbolRefAttr d : unfixedDims) {
			// ub = points(@d) -> 从 comp.dim 属性获取 (i64) -> 常量索引
			comp::DimOp dimOp = lookupDimOp(op, d);
			auto points = static_cast<int64_t>(dimOp.getPoints());

			// affine.for %iv = 0 to ub step 1
			auto forOp = mlir::affine::AffineForOp::create(rewriter, loc, /*lb=*/0, /*ub=*/points, /*step=*/1);
			loops.emplace_back(forOp);
			mlir::Value iv = forOp.getInductionVar();
			dimIndexVal[d] = iv;

			// 将插入点设置到此循环的循环体内
			rewriter.setInsertionPointToStart(forOp.getBody());
		}

		// 5) 将所有apply_init中原有的语句全部转移至最内侧循环内
		mlir::Block& srcBlock = op.getRhs().front();
		mlir::Block* dstBlock = rewriter.getInsertionBlock();
		mlir::Operation* insertPt = dstBlock->getTerminator(); // 在终结符之前插入

		// 用实际的值替换原先的临时属性
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

		// 记住内联之前的最后一个操作，即循环块顶的操作
		mlir::Operation* marker = insertPt->getPrevNode(); // 可能为 nullptr

		// 将apply_init中的操作内联到循环内
		rewriter.inlineBlockBefore(&srcBlock, insertPt, argValues);

		// 收集 coord + yield
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

		// 将插入点放在第一个非Coord操作的位置
		mlir::Operation *hoistBefore = firstNonCoord ? firstNonCoord : yieldOp.getOperation();
		rewriter.setInsertionPoint(hoistBefore);

		// 降级Coord操作
		for (comp::CoordOp c : coordsToLower) {
			mlir::Value iv = c.getIv(); // 已通过内联 argValues 替换
			mlir::Value coordVal = lowerCoord(rewriter, c.getLoc(), op, c.getDimAttr(), iv);
			if (!coordVal) return mlir::failure();
			c.replaceAllUsesWith(coordVal);
		}
		for (comp::CoordOp c : coordsToLower) {
			rewriter.eraseOp(c);
		}

		// 将插入点恢复到要创建 store 的位置（通常在 yield 位置）
		rewriter.setInsertionPoint(yieldOp);

		// 将 yield 值存储到 memref[time=0, space...]，然后删除 yield
		mlir::Value yieldedF64 = castToF64(rewriter, yieldOp.getLoc(), yieldOp.getOperand(0));

		mlir::SmallVector<mlir::Value, 8> indices;
		indices.reserve(memrefTy.getRank());
		for (auto d : dims) {
			indices.push_back(dimIndexVal.lookup(d));
		}

		mlir::memref::StoreOp::create(rewriter, yieldOp.getLoc(), yieldedF64, memref, indices);
		rewriter.eraseOp(yieldOp);

		// 6) 删除 apply_init
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

		// 标记 Affine,Memref,Arith 为合法
		target.addLegalDialect<mlir::affine::AffineDialect, mlir::memref::MemRefDialect, mlir::arith::ArithDialect>();
		// 标记 comp.for_time 为非法
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

}
