//===-- OptBoundaryHoist.cpp ---------------------------------- -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
///
//===----------------------------------------------------------------------===//


#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#include "StorePartition.h"
#include "Utils/QueryUtil.h"

namespace ezresearch {

/// 递归检查 value 的 def-use 链是否可安全 hoist：
/// 1. 遇到 mlir::memref::LoadOp 直接失败
/// 2. 有 region 的 op 保守失败
/// 3. 有副作用的 op 失败
/// 4. 其余纯 op 允许，并按“先依赖、后使用”顺序收集到 orderedOps
static bool collectHoistableDefChain(
    mlir::Value root,
    llvm::SmallVectorImpl<mlir::Operation *> &orderedOps,
    llvm::DenseSet<mlir::Operation *> &visitedOps) {

    if (llvm::isa<mlir::BlockArgument>(root)) {
        return true;
    }

    mlir::Operation *def = root.getDefiningOp();
    if (!def) {
        return true;
    }

    if (!visitedOps.insert(def).second) {
        return true;
    }

    if (llvm::isa<mlir::memref::LoadOp>(def)) {
        return false;
    }

    if (def->getNumRegions() != 0) {
        return false;
    }

    for (mlir::Value operand : def->getOperands()) {
        if (!collectHoistableDefChain(operand, orderedOps, visitedOps)) {
            return false;
        }
    }

    if (auto effectIface = llvm::dyn_cast<mlir::MemoryEffectOpInterface>(def)) {
        if (!effectIface.hasNoEffect()) {
            return false;
        }
    } else {
        if (!llvm::isa<mlir::arith::ConstantOp,
                       mlir::arith::ConstantIndexOp,
                       mlir::arith::IndexCastOp,
                       mlir::affine::AffineApplyOp>(def)) {
            return false;
        }
    }

    orderedOps.push_back(def);
    return true;
}

static bool collectHoistableDefChain(
    mlir::Value root,
    llvm::SmallVectorImpl<mlir::Operation *> &orderedOps) {
    llvm::DenseSet<mlir::Operation *> visitedOps;
    return collectHoistableDefChain(root, orderedOps, visitedOps);
}

/// 根据空间维索引重建循环 nest。
/// 约束：spatialIndices 中每个索引要么是常量，要么是 affine.for 的 IV。
static mlir::LogicalResult buildSpatialLoopNest(
    mlir::Location loc,
    llvm::ArrayRef<mlir::Value> spatialIndices,
    mlir::OpBuilder &rootBuilder,
    mlir::IRMapping &mapping,
    llvm::SmallVectorImpl<mlir::Value> &remappedSpatialIndices,
    mlir::OpBuilder &nestedBuilder) {

    nestedBuilder = mlir::OpBuilder(rootBuilder);

    for (mlir::Value idx : spatialIndices) {
        if (auto cst = getConstantIndex(idx)) {
            mlir::Value newCst = nestedBuilder.create<mlir::arith::ConstantIndexOp>(loc, *cst);
            mapping.map(idx, newCst);
            remappedSpatialIndices.push_back(newCst);
            continue;
        }

        auto loopInfo = getLoopInfoFromAffineIV(idx);
        if (!loopInfo) {
            return mlir::failure();
        }

        auto newFor = nestedBuilder.create<mlir::affine::AffineForOp>(
            loc, loopInfo->lb, loopInfo->ub, loopInfo->step);

        mlir::Value newIV = newFor.getInductionVar();
        mapping.map(idx, newIV);
        remappedSpatialIndices.push_back(newIV);

        nestedBuilder.setInsertionPointToStart(newFor.getBody());
    }

    return mlir::success();
}

/// 将已经通过检查的纯 def-use 链 clone 到 builder 当前插入点。
/// orderedOps 需要满足“先依赖、后使用”的顺序。
static void cloneDefChainWithMapping(
    llvm::ArrayRef<mlir::Operation *> orderedOps,
    mlir::OpBuilder &builder,
    mlir::IRMapping &mapping) {

    for (mlir::Operation *op : orderedOps) {
        mlir::Operation *cloned = builder.clone(*op, mapping);
        mapping.map(op, cloned);

        for (auto [oldRes, newRes] :
             llvm::zip(op->getResults(), cloned->getResults())) {
            mapping.map(oldRes, newRes);
        }
    }
}

/// 生成 hoisted store：time = 0 / 1
static void createHoistedStores(
    mlir::memref::StoreOp oldStore,
    mlir::Value newStoredValue,
    llvm::ArrayRef<mlir::Value> remappedSpatialIndices,
    mlir::OpBuilder &builder) {

    mlir::Location loc = oldStore.getLoc();
    mlir::Value memref = oldStore.getMemRef();

    mlir::Value t0 = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value t1 = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);

    llvm::SmallVector<mlir::Value> idx0;
    llvm::SmallVector<mlir::Value> idx1;

    idx0.push_back(t0);
    idx1.push_back(t1);

    idx0.append(remappedSpatialIndices.begin(), remappedSpatialIndices.end());
    idx1.append(remappedSpatialIndices.begin(), remappedSpatialIndices.end());

    builder.create<mlir::memref::StoreOp>(loc, newStoredValue, memref, idx0);
    builder.create<mlir::memref::StoreOp>(loc, newStoredValue, memref, idx1);
}

struct OptBoundaryHoistPass : public mlir::PassWrapper<OptBoundaryHoistPass, mlir::OperationPass<mlir::ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OptBoundaryHoistPass)

    llvm::StringRef getArgument() const final { return "boundary-hoist"; }

    void runOnOperation() override {
        mlir::ModuleOp module = getOperation();

        llvm::DenseMap<mlir::Value, StorePartition> allocMap;

        module.walk([&](mlir::Operation *op) {
            if (auto alloc = llvm::dyn_cast<mlir::memref::AllocOp>(op)) {
                mlir::Value memrefHandle = alloc.getResult();
                allocMap.try_emplace(memrefHandle, alloc);
                return;
            }

            if (auto store = llvm::dyn_cast<mlir::memref::StoreOp>(op)) {
                mlir::Value targetMemref = store.getMemRef();

                auto &sp = allocMap.find(targetMemref)->second;
                sp.emplace(store);
            }
        });

        llvm::SmallVector<mlir::Operation *> eraseList;
        for (auto &kv : allocMap) {
            StorePartition &sp = kv.second;
            auto hoistStores = sp.analyze();

            for (mlir::memref::StoreOp storeOp : hoistStores) {
                llvm::SmallVector<mlir::Value> indices(storeOp.getIndices().begin(),
                                                       storeOp.getIndices().end());
                if (indices.empty()) {
                    continue;
                }

                mlir::Value timeIdx = indices.front();
                mlir::Value storedValue = storeOp.getValue();

                auto timeLoopInfo = getLoopInfoFromAffineIV(timeIdx);
                if (!timeLoopInfo) {
                    continue;
                }

                mlir::affine::AffineForOp timeFor = timeLoopInfo->for_op;

                llvm::SmallVector<mlir::Operation *> defChain;
                if (!collectHoistableDefChain(storedValue, defChain)) {
                    continue;
                }

                mlir::OpBuilder preBuilder(timeFor);
                mlir::IRMapping mapping;

                llvm::SmallVector<mlir::Value> spatialIndices(indices.begin() + 1,
                                                              indices.end());
                llvm::SmallVector<mlir::Value> remappedSpatialIndices;
                mlir::OpBuilder nestedBuilder = preBuilder;

                if (mlir::failed(buildSpatialLoopNest(storeOp.getLoc(),
                                                      spatialIndices,
                                                      preBuilder,
                                                      mapping,
                                                      remappedSpatialIndices,
                                                      nestedBuilder))) {
                    continue;
                }

                cloneDefChainWithMapping(defChain, nestedBuilder, mapping);

                mlir::Value newStoredValue = mapping.lookupOrDefault(storedValue);
                if (!newStoredValue) {
                    newStoredValue = storedValue;
                }

                createHoistedStores(storeOp,
                                    newStoredValue,
                                    remappedSpatialIndices,
                                    nestedBuilder);

                eraseList.push_back(storeOp.getOperation());
            }
        }

        for (mlir::Operation *op : eraseList) {
            op->erase();
        }
    }
};

void registerOptBoundaryHoistPass() {
    mlir::PassRegistration<OptBoundaryHoistPass>();
}

std::unique_ptr<mlir::Pass> createOptBoundaryHoistPass() {
    return std::make_unique<OptBoundaryHoistPass>();
}

}
