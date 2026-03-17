//===-- OptLoopTiling.cpp ------------------------------------- -*- C++ -*-===//
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


#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <hwloc.h>

#include "BuildTilingNest.h"
#include "TilingSizeSolve.h"
#include "Utils/CacheUtil.h"
#include "Utils/QueryUtil.h"

namespace ezresearch {

constexpr uint64_t E = 8;

struct OptLoopTilingPass : public mlir::PassWrapper<OptLoopTilingPass, mlir::OperationPass<mlir::ModuleOp> > {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OptLoopTilingPass)

    llvm::StringRef getArgument() const final { return "loop-tiling"; }

    void runOnOperation() override {
        uint64_t CL;
        uint64_t C;
        getTilingCacheParams(CL, C);

        auto B = CL / E; // B表示每条缓存行内的元素数
        auto M = C / E; // M表示缓存中可容纳的元素数

        auto module = getOperation();
        std::vector<LoopInfo> loop_infos;
        std::set<LoopInfo> vis;

        module.walk([&](mlir::Operation *op) {
            auto collectFromIndices = [&](mlir::AffineMap map,mlir::Operation::operand_range operands) {
                // 跳过第一维，从第二维开始
                for (size_t i = 1; i < map.getNumResults(); ++i) {
                    mlir::AffineExpr expr = map.getResult(i);
                    expr.walk([&](mlir::AffineExpr subExpr) {
                        if (auto d = llvm::dyn_cast<mlir::AffineDimExpr>(subExpr)) {
                            mlir::Value v = operands[d.getPosition()];
                            if (auto loop_info = getLoopInfoFromAffineIV(v)) {
                                if (vis.find(*loop_info) == vis.end()) {
                                    loop_infos.push_back(*loop_info);
                                    vis.insert(*loop_info);
                                }
                            }
                        }
                    });
                }
            };

            if (auto loadOp = llvm::dyn_cast<mlir::affine::AffineLoadOp>(op)) {
                collectFromIndices(loadOp.getAffineMap(),loadOp.getMapOperands());
                return;
            }

            if (auto storeOp = llvm::dyn_cast<mlir::affine::AffineStoreOp>(op)) {
                collectFromIndices(storeOp.getAffineMap(), storeOp.getMapOperands());
                return;
            }
        });

        //如果一行的东西太少了，就不分块
        auto line = loop_infos.back().ub - loop_infos.back().lb;
        if (M > 4 * line) {
            return ;
        }

        auto tiling_size = TilingSizeSolve(loop_infos, B, M);

        // 1. 按 loop_infos 顺序抽出 for band
        llvm::SmallVector<mlir::affine::AffineForOp, 8> band;
        band.reserve(loop_infos.size());
        for (auto &info : loop_infos) {
            band.push_back(info.for_op);
        }

        // 2. 检查 band 是否可以做 tiling
        if (!mlir::affine::isTilingValid(band)) {
            module.emitError("selected affine loop band is not valid for tiling");
            signalPassFailure();
            return;
        }

        // 3. 准备 tile sizes
        llvm::SmallVector<unsigned, 8> tileSizes;
        tileSizes.reserve(tiling_size.size());
        for (size_t i = 0; i < tiling_size.size(); ++i) {
            uint64_t ts = tiling_size[i];
            uint64_t tripCount = loop_infos[i].ub - loop_infos[i].lb;
            // 防止非法 tile size
            if (ts == 0) {
                ts = 1;
            }
            // tile size 不能超过循环长度
            if (ts >= tripCount) {
                ts = 1;
            }

            tileSizes.push_back(static_cast<unsigned>(ts));
        }


        if (tileSizes.empty()) {
            return;
        }

        // 4. 执行 tiling
        llvm::SmallVector<mlir::affine::AffineForOp, 8> generatedCases;
        if (mlir::failed(buildTilingNest(loop_infos, tileSizes, &generatedCases))) {
            module.emitError("buildTiledNest failed");
            signalPassFailure();
            return;
        }
    }
};

void registerOptLoopTilingPass() {
    mlir::PassRegistration<OptLoopTilingPass>();
}

std::unique_ptr<mlir::Pass> createOptLoopTilingPass() {
    return std::make_unique<OptLoopTilingPass>();
}

}
