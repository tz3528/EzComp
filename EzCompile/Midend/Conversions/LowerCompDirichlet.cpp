//===-- LowerCompDirichlet.cpp ---------------------------------*- C++ -*-===//
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
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Dialects/Comp/include/Comp.h"
#include "Utils/BuilderUtil.h"
#include "Utils/LowerUtil.h"

namespace ezcompile {

struct LowerDirichletPattern : mlir::OpConversionPattern<comp::DirichletOp> {
	using OpConversionPattern<comp::DirichletOp>::OpConversionPattern;

	mlir::LogicalResult matchAndRewrite(comp::DirichletOp op,
	                              OpAdaptor adaptor,
	                              mlir::ConversionPatternRewriter& rewriter) const override {
		mlir::Location loc = op.getLoc();
		mlir::Block& srcBlock = op.getRhs().front();

		// 1. 收集memref、for_time和update
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

		comp::ForTimeOp for_time;
		comp::UpdateOp update;

		for (mlir::Operation *user : field.getUsers()) {
			if (auto u = mlir::dyn_cast<comp::UpdateOp>(user)) {
				update = u;
				break;
			}
		}
		for_time = update->getParentOfType<comp::ForTimeOp>();

		// 2. 分析和存储锚点信息及属性
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

		// memref 的秩应为 1(time)+N(space)
		if (unfixedDims.size() + fixed.size() != memrefTy.getRank()) {
			op.emitError() << "space dims count (" << unfixedDims.size()
				<< ") does not match memref rank (" << (memrefTy.getRank() - 1) << ")";
			return mlir::failure();
		}

		// =========================================================================
        // 核心辅助函数：在当前插入点克隆并生成计算逻辑
        // 使用 clone 而不是 inline，解决 double-free 问题
        // =========================================================================
        auto emitCalculationLogic = [&](mlir::DenseMap<mlir::Attribute, mlir::Value>& currentDimIndices) -> mlir::LogicalResult {
            mlir::IRMapping mapper; // 用于映射原 Block 的值到新克隆的值

            // A. 映射 Block Arguments (Coord getIv() 获取的值)
            // CoordOp 读取的是 Block 参数，我们需要将其映射到当前的循环 IV 或固定值
            for (mlir::Operation& inner : srcBlock.getOperations()) {
                if (auto coord = dyn_cast<comp::CoordOp>(inner)) {
                    mlir::Value iv = coord.getIv();
                    if (auto barg = dyn_cast<mlir::BlockArgument>(iv)) {
                        if (barg.getOwner() == &srcBlock) {
                            auto dimAttr = coord.getDimAttr();
                            // 查找当前上下文中的维度值（可能是 IV，也可能是常量）
                            if (currentDimIndices.count(dimAttr)) {
                                mapper.map(barg, currentDimIndices[dimAttr]);
                            }
                        }
                    }
                }
            }

            // B. 克隆 Op (跳过 CoordOp 和 YieldOp 的特殊处理)
            // 注意：我们不移动原 Op，而是克隆它们
            for (mlir::Operation &opInst : srcBlock.without_terminator()) {
                if (auto coord = dyn_cast<comp::CoordOp>(opInst)) {
                    // 特殊处理 CoordOp：不要克隆它，而是计算出它的值
                    mlir::Value mappedIv = mapper.lookupOrDefault(coord.getIv());

                    // 调用你的 lowerCoord 函数生成计算代码 (Arith Ops)
                    mlir::Value coordVal = lowerCoord(rewriter, coord.getLoc(), op, coord.getDimAttr(), mappedIv);
                    if (!coordVal) return mlir::failure();

                    // 将原 CoordOp 的结果映射到新计算的值
                    mapper.map(coord.getResult(), coordVal);
                } else {
                    // 普通 Op：直接克隆
                    rewriter.clone(opInst, mapper);
                }
            }

            // C. 处理 Yield (Terminator)
            if (auto yieldOp = dyn_cast<comp::YieldOp>(srcBlock.getTerminator())) {
                if (yieldOp.getNumOperands() != 1) return mlir::failure();

                // 获取 yield 的值 (通过 mapper 解析)
                mlir::Value yieldedVal = mapper.lookupOrDefault(yieldOp.getOperand(0));
                mlir::Value yieldedF64 = castToF64(rewriter, yieldOp.getLoc(), yieldedVal);

                // 准备 Store 的下标
                mlir::SmallVector<mlir::Value, 8> storeIndices;
                for (auto d : dims) {
                    storeIndices.push_back(currentDimIndices.lookup(d));
                }

                // 生成 Store
                rewriter.create<mlir::memref::StoreOp>(yieldOp.getLoc(), yieldedF64, memref, storeIndices);
            }
            return mlir::success();
        };

        // 辅助递归函数：构建空间循环嵌套
        std::function<mlir::LogicalResult(int, mlir::DenseMap<mlir::Attribute, mlir::Value>&)> buildLoopNest =
            [&](int dimIdx, mlir::DenseMap<mlir::Attribute, mlir::Value>& indicesMap) -> mlir::LogicalResult {
            if (dimIdx >= unfixedDims.size()) {
                return emitCalculationLogic(indicesMap);
            }

            mlir::FlatSymbolRefAttr d = unfixedDims[dimIdx];
            comp::DimOp dimOp = lookupDimOp(op, d);
            auto points = static_cast<int64_t>(dimOp.getPoints());

            auto forOp = rewriter.create<mlir::affine::AffineForOp>(loc, 0, points, 1);
            indicesMap[d] = forOp.getInductionVar(); // 记录当前维度的 IV

            rewriter.setInsertionPointToStart(forOp.getBody());
            return buildLoopNest(dimIdx + 1, indicesMap);
        };

		// 3. 第一处插入：ForTime 之前 (初始化边界)
        {
            rewriter.setInsertionPoint(for_time);
            mlir::DenseMap<mlir::Attribute, mlir::Value> dimIndexVal;

            // 固定维度
            for (auto& kv : fixed) {
                if (auto d = dyn_cast<mlir::FlatSymbolRefAttr>(kv.first))
                    dimIndexVal[d] = constIndex(rewriter, loc, kv.second);
            }

            if(time_var) dimIndexVal[time_var] = constIndex(rewriter, loc, 0);

            if (mlir::failed(buildLoopNest(0, dimIndexVal))) return mlir::failure();
        }

        // 4. 第二处插入：ForTime 内部 (更新边界)
        {
            mlir::Block *bodyBlock = &for_time.getRegion().front();
            rewriter.setInsertionPointToEnd(bodyBlock);

            mlir::DenseMap<mlir::Attribute, mlir::Value> dimIndexVal;

            // 固定维度
            for (auto& kv : fixed) {
                if (auto d = dyn_cast<mlir::FlatSymbolRefAttr>(kv.first))
                    dimIndexVal[d] = constIndex(rewriter, loc, kv.second);
            }

			mlir::Value timeIV = bodyBlock->getArgument(0);

            dimIndexVal[time_var] = modIndex(rewriter, loc, timeIV, 2);

            if (mlir::failed(buildLoopNest(0, dimIndexVal))) return mlir::failure();
        }

		// 5. 删除原操作
		rewriter.eraseOp(op);
		return mlir::success();
	}
};

struct LowerCompDirichletPass : mlir::PassWrapper<LowerCompDirichletPass, mlir::OperationPass<mlir::ModuleOp>> {
	MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerCompDirichletPass)

	void getDependentDialects(mlir::DialectRegistry& registry) const override {
		registry.insert<mlir::arith::ArithDialect, mlir::memref::MemRefDialect, comp::CompDialect>();
	}

	void runOnOperation() override {
		mlir::MLIRContext* context = &getContext();
		mlir::ModuleOp module = getOperation();

		mlir::ConversionTarget target(*context);

		// 标记 Affine, Arith 为合法
		target.addLegalDialect<mlir::affine::AffineDialect, mlir::memref::MemRefDialect, mlir::arith::ArithDialect>();
		// 标记 comp.for_time 为非法，强制框架对其进行转换
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

}
