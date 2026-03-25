//===-- AdjustTimeIndex.cpp ----------------------------------- -*- C++ -*-===//
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


#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"

namespace ezcompile {

static mlir::Value buildMod2Index(mlir::OpBuilder &builder, mlir::Location loc,
                                  mlir::Value idx) {
    mlir::Value c2 = builder.create<mlir::arith::ConstantIndexOp>(loc, 2);
    return builder.create<mlir::arith::RemUIOp>(loc, idx, c2);
}

static mlir::AffineMap adjustFirstResultMod2(mlir::AffineMap oldMap,
                                             mlir::MLIRContext *ctx) {
    mlir::SmallVector<mlir::AffineExpr> results(oldMap.getResults().begin(),
                                                oldMap.getResults().end());
    if (results.empty()) {
        return oldMap;
    }

    results[0] = results[0] % 2;
    return mlir::AffineMap::get(oldMap.getNumDims(), oldMap.getNumSymbols(),
                                results, ctx);
}

struct AdjustTimeIndexPass : public mlir::PassWrapper<AdjustTimeIndexPass, mlir::OperationPass<mlir::ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AdjustTimeIndexPass)

    mlir::StringRef getArgument() const final {
        return "adjust-time-index";
    }

    mlir::StringRef getDescription() const final {
        return "Adjust first index of memref ops and first result of affine maps by mod 2";
    }

    void runOnOperation() override {
        mlir::ModuleOp module = getOperation();
        mlir::MLIRContext *ctx = &getContext();

        mlir::SmallVector<mlir::Operation *> ops;
        module.walk([&](mlir::Operation *op) {
            if (mlir::isa<mlir::memref::LoadOp, mlir::memref::StoreOp,
                          mlir::affine::AffineLoadOp, mlir::affine::AffineStoreOp>(op)) {
                ops.push_back(op);
            }
        });

        for (mlir::Operation *op : ops) {
            mlir::OpBuilder builder(op);
            mlir::Location loc = op->getLoc();

            if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(op)) {
                mlir::SmallVector<mlir::Value> indices(load.getIndices().begin(), load.getIndices().end());
                if (indices.empty()) {
                    continue;
                }

                indices[0] = buildMod2Index(builder, loc, indices[0]);

                auto newLoad = builder.create<mlir::memref::LoadOp>(loc, load.getMemRef(), indices);
                load.replaceAllUsesWith(newLoad.getResult());
                load.erase();
                continue;
            }

            if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(op)) {
                mlir::SmallVector<mlir::Value> indices(store.getIndices().begin(), store.getIndices().end());
                if (indices.empty()) {
                    continue;
                }

                indices[0] = buildMod2Index(builder, loc, indices[0]);

                builder.create<mlir::memref::StoreOp>(loc, store.getValue(), store.getMemRef(), indices);
                store.erase();
                continue;
            }

            if (auto load = mlir::dyn_cast<mlir::affine::AffineLoadOp>(op)) {
                mlir::AffineMap newMap = adjustFirstResultMod2(load.getAffineMap(), ctx);

                auto newLoad = builder.create<mlir::affine::AffineLoadOp>(
                    loc, load.getMemRef(), newMap, load.getMapOperands());
                load.replaceAllUsesWith(newLoad.getResult());
                load.erase();
                continue;
            }

            if (auto store = mlir::dyn_cast<mlir::affine::AffineStoreOp>(op)) {
                mlir::AffineMap newMap = adjustFirstResultMod2(store.getAffineMap(), ctx);

                builder.create<mlir::affine::AffineStoreOp>(
                    loc, store.getValueToStore(), store.getMemRef(), newMap, store.getMapOperands());
                store.erase();
                continue;
            }
        }
    }
};

void registerAdjustTimeIndexPass() {
    mlir::PassRegistration<AdjustTimeIndexPass>();
}

std::unique_ptr<mlir::Pass> createAdjustTimeIndexPass() {
    return std::make_unique<AdjustTimeIndexPass>();
}

}
