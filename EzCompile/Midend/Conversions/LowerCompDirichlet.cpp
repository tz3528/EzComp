//===-- LowerCompDirichlet.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// comp.dirichlet 降级实现
// 将 Dirichlet 边界条件降级为两处赋值：
// - ForTime 之前：初始化边界
/// - ForTime 内部：每时间步更新边界
//
//===----------------------------------------------------------------------===//


#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Dialects/Comp/include/Comp.h"
#include "Utils/BuilderUtil.h"
#include "Utils/LowerUtil.h"

namespace ezcompile {

/// 降级 Pattern：将 comp.dirichlet 转换为边界赋值代码
///
/// 实现思路：
/// 1. 解析锚点，区分固定维度和自由维度
/// 2. 在 ForTime 之前插入初始边界赋值（time=0）
/// 3. 在 ForTime 内部插入每时间步的边界更新
/// 4. 使用 clone 而非 inline 避免重复释放问题
struct LowerDirichletPattern : mlir::OpConversionPattern<comp::DirichletOp> {
	using OpConversionPattern<comp::DirichletOp>::OpConversionPattern;

	mlir::LogicalResult matchAndRewrite(comp::DirichletOp op,
	                              OpAdaptor adaptor,
	                              mlir::ConversionPatternRewriter& rewriter) const override {
		mlir::Location loc = op.getLoc();
		mlir::Block& srcBlock = op.getRhs().front();

		mlir::memref::AllocOp alloc = getDefiningFieldOp(op.getField());
		if (!alloc) {
			return rewriter.notifyMatchFailure(op, "field is not backed by memref.alloc");
		}
		mlir::Value memref = alloc.getResult();
		auto memrefTy = dyn_cast<mlir::MemRefType>(memref.getType());
		if (!memrefTy) {
			return rewriter.notifyMatchFailure(op, "field alloc is not a memref");
		}

		// 查找关联的 ForTime 和 Update
		comp::ForTimeOp for_time;
		comp::UpdateOp update;

		for (mlir::Operation *user : op.getField().getUsers()) {
			if (auto u = mlir::dyn_cast<comp::UpdateOp>(user)) {
				update = u;
				break;
			}
		}
		for_time = update->getParentOfType<comp::ForTimeOp>();

		// 解析锚点
		mlir::DenseMap<mlir::Attribute, uint64_t> fixed;
		for (mlir::Attribute a : op.getAnchors()) {
			auto anc = dyn_cast<comp::AnchorAttr>(a);
			if (!anc) continue;
			fixed[anc.getDim()] = anc.getIndex();
		}

		mlir::SmallVector<mlir::FlatSymbolRefAttr, 4> unfixedDims;
		mlir::SmallVector<mlir::FlatSymbolRefAttr, 4> dims;
		mlir::Attribute time_var;
		mlir::Operation* symTableOp = op->getParentOfType<comp::ProblemOp>();
		if (!symTableOp) {
			return rewriter.notifyMatchFailure(op, "cannot find symbol table for dim ordering");
		}
		symTableOp->walk([&](comp::DimOp dim) {
			mlir::FlatSymbolRefAttr sym = mlir::FlatSymbolRefAttr::get(dim.getSymNameAttr());
			if (dim.getTimeVar()) time_var = sym;
			if (!fixed.contains(sym)) unfixedDims.emplace_back(sym);
			dims.emplace_back(sym);
		});

		if (unfixedDims.size() + fixed.size() != memrefTy.getRank()) {
			op.emitError() << "space dims count (" << unfixedDims.size()
				<< ") does not match memref rank (" << (memrefTy.getRank() - 1) << ")";
			return mlir::failure();
		}

		// 核心函数：克隆并生成计算逻辑
        auto emitCalculationLogic = [&](mlir::DenseMap<mlir::Attribute, mlir::Value>& currentDimIndices) -> mlir::LogicalResult {
            mlir::IRMapping mapper;

            // 映射 Block Arguments
            for (mlir::Operation& inner : srcBlock.getOperations()) {
                if (auto coord = dyn_cast<comp::CoordOp>(inner)) {
                    mlir::Value iv = coord.getIv();
                    if (auto barg = dyn_cast<mlir::BlockArgument>(iv)) {
                        if (barg.getOwner() == &srcBlock) {
                            auto dimAttr = coord.getDimAttr();
                            if (currentDimIndices.count(dimAttr)) {
                                mapper.map(barg, currentDimIndices[dimAttr]);
                            }
                        }
                    }
                }
            }

            // 克隆 Op（coord 特殊处理）
            for (mlir::Operation &opInst : srcBlock.without_terminator()) {
                if (auto coord = dyn_cast<comp::CoordOp>(opInst)) {
                    mlir::Value mappedIv = mapper.lookupOrDefault(coord.getIv());
                    mlir::Value coordVal = lowerCoord(rewriter, coord.getLoc(), op, coord.getDimAttr(), mappedIv);
                    if (!coordVal) return mlir::failure();
                    mapper.map(coord.getResult(), coordVal);
                } else {
                    rewriter.clone(opInst, mapper);
                }
            }

            // 处理 Yield
            if (auto yieldOp = dyn_cast<comp::YieldOp>(srcBlock.getTerminator())) {
                if (yieldOp.getNumOperands() != 1) return mlir::failure();

                mlir::Value yieldedVal = mapper.lookupOrDefault(yieldOp.getOperand(0));
                mlir::Value yieldedF64 = castToF64(rewriter, yieldOp.getLoc(), yieldedVal);

                mlir::SmallVector<mlir::Value, 8> storeIndices;
                for (auto d : dims) {
                    storeIndices.push_back(currentDimIndices.lookup(d));
                }

                rewriter.create<mlir::memref::StoreOp>(yieldOp.getLoc(), yieldedF64, memref, storeIndices);
            }
            return mlir::success();
        };

		// 递归构建循环嵌套
		std::function<mlir::LogicalResult(int, mlir::DenseMap<mlir::Attribute, mlir::Value>&)> buildLoopNest =
			[&](int dimIdx, mlir::DenseMap<mlir::Attribute, mlir::Value>& indicesMap) -> mlir::LogicalResult {
			if (dimIdx >= unfixedDims.size()) {
				return emitCalculationLogic(indicesMap);
			}

			mlir::FlatSymbolRefAttr d = unfixedDims[dimIdx];
			comp::DimOp dimOp = lookupDimOp(op, d);
			auto points = static_cast<int64_t>(dimOp.getPoints());

			if (d != time_var) {
				auto forOp = rewriter.create<mlir::affine::AffineForOp>(loc, 0, points, 1);
				rewriter.setInsertionPointToStart(forOp.getBody());
				indicesMap[d] = forOp.getInductionVar();
			}

			return buildLoopNest(dimIdx + 1, indicesMap);
		};
		// 第一处：ForTime 之前（初始化边界）
        {
            rewriter.setInsertionPoint(for_time);
            mlir::DenseMap<mlir::Attribute, mlir::Value> dimIndexVal;

            for (auto& kv : fixed) {
                if (auto d = dyn_cast<mlir::FlatSymbolRefAttr>(kv.first)) {
	                dimIndexVal[d] = constIndex(rewriter, loc, kv.second);
                }
            }

            if(time_var) dimIndexVal[time_var] = constIndex(rewriter, loc, 0);

            if (mlir::failed(buildLoopNest(0, dimIndexVal))) return mlir::failure();
        }

        // 第二处：ForTime 内部（更新边界）
        {
            mlir::Block *bodyBlock = &for_time.getRegion().front();
            rewriter.setInsertionPointToEnd(bodyBlock);

            mlir::DenseMap<mlir::Attribute, mlir::Value> dimIndexVal;

            for (auto& kv : fixed) {
                if (auto d = dyn_cast<mlir::FlatSymbolRefAttr>(kv.first)) {
	                dimIndexVal[d] = constIndex(rewriter, loc, kv.second);
                }
            }

			mlir::Value timeIV = bodyBlock->getArgument(0);
			mlir::Value timeIVPlusOne = rewriter.create<mlir::arith::AddIOp>(
					loc, timeIV, constIndex(rewriter, loc, 1));
			dimIndexVal[time_var] = timeIVPlusOne;

            if (mlir::failed(buildLoopNest(0, dimIndexVal))) return mlir::failure();
        }

		rewriter.eraseOp(op);
		return mlir::success();
	}
};

struct LowerCompDirichletPass : mlir::PassWrapper<LowerCompDirichletPass, mlir::OperationPass<mlir::ModuleOp>> {
	MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerCompDirichletPass)

	void getDependentDialects(mlir::DialectRegistry& registry) const override {
		registry.insert<mlir::arith::ArithDialect, mlir::memref::MemRefDialect, comp::CompDialect, mlir::math::MathDialect>();
	}

	mlir::StringRef getArgument() const override { return "lower-comp-dirichlet"; }

	void runOnOperation() override {
		mlir::MLIRContext* context = &getContext();
		mlir::ModuleOp module = getOperation();

		mlir::ConversionTarget target(*context);

		target.addLegalDialect<mlir::affine::AffineDialect, mlir::memref::MemRefDialect, mlir::arith::ArithDialect, mlir::math::MathDialect>();
		target.addIllegalOp<comp::DirichletOp>();

		mlir::RewritePatternSet patterns(context);
		patterns.add<LowerDirichletPattern>(context);

		if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
			signalPassFailure();
		}
	}
};

void registerLowerCompDirichletPass() {
	mlir::PassRegistration<LowerCompDirichletPass>();
}

std::unique_ptr<mlir::Pass> createLowerCompDirichletPass() {
	return std::make_unique<LowerCompDirichletPass>();
}

} // namespace ezcompile