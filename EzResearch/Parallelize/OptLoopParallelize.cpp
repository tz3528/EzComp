//===-- OptLoopParallelize.cpp -------------------------------- -*- C++ -*-===//
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


#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace ezresearch {

struct OptLoopParallelizePass : public mlir::PassWrapper<OptLoopParallelizePass, mlir::OperationPass<mlir::ModuleOp> > {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OptLoopParallelizePass)

    llvm::StringRef getArgument() const final { return "loop-parallelize"; }

    void runOnOperation() override {
        mlir::ModuleOp module = getOperation();

        std::set<mlir::affine::AffineForOp> legal_for;
        module.walk([&](mlir::affine::AffineForOp forOp) {
            llvm::SmallVector<mlir::affine::LoopReduction> reductions;
            if (mlir::affine::isLoopParallel(forOp, &reductions)) {
                legal_for.insert(forOp);
            }
        });

        std::map<mlir::affine::AffineForOp, mlir::affine::AffineForOp> fa;
        std::map<mlir::affine::AffineForOp, std::vector<mlir::affine::AffineForOp>> son;
        std::set<mlir::affine::AffineForOp> root;
        std::map<mlir::affine::AffineForOp, uint64_t> subtreeWork;
        std::map<mlir::affine::AffineForOp, uint64_t> tripCount;
        std::map<mlir::affine::AffineForOp, bool> selected;
        for (auto for_op : legal_for) {
            auto parent = for_op->getParentOfType<mlir::affine::AffineForOp>();
            if (parent == nullptr) {
                root.insert(for_op);
            }
            else if (legal_for.find(parent) != legal_for.end()) {
                fa[for_op] = parent;
                son[parent].push_back(for_op);
            }
            else {
                root.insert(for_op);
            }

            if (auto tc = mlir::affine::getConstantTripCount(for_op)) {
                tripCount[for_op] = *tc;
            } else {
                // 拿不到常数 trip count 的兜底逻辑
                tripCount[for_op] = 1;
            }
        }

        auto dfs = [&](mlir::affine::AffineForOp u, auto dfs) -> void {
            uint64_t childWorkSum = 0;
            for (auto v : son[u]) {
                dfs(v, dfs);
                childWorkSum += subtreeWork[v];
            }

            uint64_t tc = 1;
            auto it = tripCount.find(u);
            if (it != tripCount.end()) {
                tc = std::max<uint64_t>(1, it->second);
            }

            if (son[u].empty()) {
                // 叶子：认为子循环的工作量为 1
                subtreeWork[u] = tc;
            } else {
                subtreeWork[u] = tc * std::max<uint64_t>(1, childWorkSum);
            }

            selected[u] = (tc >= 32 && subtreeWork[u] >= 10000);
        };

        for (auto for_op : root) {
            dfs(for_op, dfs);
        }

        for (auto it : selected) {
            auto for_op = it.first;
            auto flag = it.second;
            if (flag) {
                mlir::affine::AffineParallelOp parOp;
                if (failed(mlir::affine::affineParallelize(for_op, /*parallelReductions=*/{}, &parOp))) {
                    for_op->emitRemark() << "parallelize failed";
                }
            }
        }
    }
};

void registerOptLoopParallelizePass() {
    mlir::PassRegistration<OptLoopParallelizePass>();
}

std::unique_ptr<mlir::Pass> createOptLoopParallelizePass() {
    return std::make_unique<OptLoopParallelizePass>();
}

}
