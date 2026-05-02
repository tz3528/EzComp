//===-- BuildTilingNest.cpp ----------------------------------- -*- C++ -*-===//
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


#include "BuildTilingNest.h"

namespace ezresearch {

struct TileDimInfo {
    int64_t fullUb;      // [lb, fullUb) 是 full-tile 区
    int64_t tileSize;
    bool hasFull;
    bool hasTail;
};

static void collectInnermostBodyOps(
    mlir::affine::AffineForOp innermost,
    llvm::SmallVectorImpl<mlir::Operation *> &bodyOps) {
    bodyOps.clear();
    for (mlir::Operation &op : innermost.getBody()->without_terminator()) {
        bodyOps.push_back(&op);
    }
}

static llvm::SmallVector<TileDimInfo, 8> computeTileDimInfos(
    std::vector<LoopInfo> loopInfos,
    llvm::ArrayRef<unsigned> tileSizes) {
    llvm::SmallVector<TileDimInfo, 8> dims;
    dims.reserve(loopInfos.size());

    for (size_t i = 0; i < loopInfos.size(); ++i) {
        int64_t trip = loopInfos[i].ub - loopInfos[i].lb;
        int64_t tile = static_cast<int64_t>(tileSizes[i]);
        int64_t fullTrip = (trip / tile) * tile;
        int64_t fullUb = loopInfos[i].lb + fullTrip;

        dims.push_back(TileDimInfo{
            fullUb,
            tile,
            fullUb > loopInfos[i].lb,
            fullUb < loopInfos[i].ub
        });
    }

    return dims;
}

static mlir::AffineMap getIdentityMap(mlir::MLIRContext *ctx) {
    auto d0 = mlir::getAffineDimExpr(0, ctx);
    return mlir::AffineMap::get(1, 0, d0);
}

static mlir::AffineMap getPlusConstMap(mlir::MLIRContext *ctx, int64_t c) {
    auto d0 = mlir::getAffineDimExpr(0, ctx);
    auto cst = mlir::getAffineConstantExpr(c, ctx);
    return mlir::AffineMap::get(1, 0, d0 + cst);
}

static mlir::affine::AffineForOp createStaticFor(
    mlir::OpBuilder &builder, mlir::Location loc,
    int64_t lb, int64_t ub, int64_t step) {
    return builder.create<mlir::affine::AffineForOp>(loc, lb, ub, step);
}

static mlir::affine::AffineForOp createTilePointFor(
    mlir::OpBuilder &builder, mlir::Location loc,
    mlir::Value tileIv, int64_t tileSize, int64_t step) {
    auto idMap = getIdentityMap(builder.getContext());
    auto ubMap = getPlusConstMap(builder.getContext(), tileSize);

    return builder.create<mlir::affine::AffineForOp>(
        loc,
        mlir::ValueRange{tileIv}, idMap,
        mlir::ValueRange{tileIv}, ubMap,
        step);
}

static mlir::LogicalResult cloneBody(
    mlir::OpBuilder &builder,
    std::vector<LoopInfo>& loopInfos,
    llvm::ArrayRef<mlir::Value> newPointIvs,
    llvm::ArrayRef<mlir::Operation *> bodyOps) {
    mlir::IRMapping mapper;
    for (size_t i = 0; i < loopInfos.size(); ++i) {
        mapper.map(loopInfos[i].for_op.getInductionVar(), newPointIvs[i]);
    }

    for (mlir::Operation *op : bodyOps) {
        builder.clone(*op, mapper);
    }

    return mlir::success();
}

static bool shouldBuildCase(llvm::ArrayRef<TileDimInfo> dims, uint64_t mask) {
    for (size_t i = 0; i < dims.size(); ++i) {
        bool useTail = ((mask >> i) & 1ULL) != 0;
        if (useTail && !dims[i].hasTail)
            return false;
        if (!useTail && !dims[i].hasFull)
            return false;
    }
    return true;
}

static mlir::LogicalResult buildOneCase(
    mlir::OpBuilder &builder,
    mlir::Location loc,
    std::vector<LoopInfo> loopInfos,
    llvm::ArrayRef<TileDimInfo> dims,
    uint64_t mask,
    llvm::ArrayRef<mlir::Operation *> bodyOps,
    mlir::affine::AffineForOp &rootCaseLoop) {

    rootCaseLoop = nullptr;

    llvm::SmallVector<mlir::Value, 8> outerTileIvs(loopInfos.size());
    llvm::SmallVector<bool, 8> isFullDim;
    isFullDim.reserve(loopInfos.size());

    // 第一阶段：只建 full-tile 的外层 tile loop
    for (size_t i = 0; i < loopInfos.size(); ++i) {
        const auto &loop = loopInfos[i];
        const auto &dim = dims[i];
        bool useTail = ((mask >> i) & 1ULL) != 0;

        if (!useTail) {
            auto tileLoop =
                createStaticFor(builder, loc, loop.lb, dim.fullUb, dim.tileSize);
            if (!rootCaseLoop)
                rootCaseLoop = tileLoop;

            outerTileIvs[i] = tileLoop.getInductionVar();
            isFullDim.push_back(true);

            builder.setInsertionPointToStart(tileLoop.getBody());
        } else {
            // tail 维度不在第一阶段生成
            outerTileIvs[i] = mlir::Value();
            isFullDim.push_back(false);
        }
    }

    // 第二阶段：按维度生成真正执行 body 的 point/tail loops
    llvm::SmallVector<mlir::Value, 8> pointIvs;
    pointIvs.reserve(loopInfos.size());

    for (size_t i = 0; i < loopInfos.size(); ++i) {
        const auto &loop = loopInfos[i];
        const auto &dim = dims[i];

        if (isFullDim[i]) {
            // full case: tile iv -> point loop
            auto pointLoop = createTilePointFor(
                builder, loc, outerTileIvs[i], dim.tileSize, loop.step);
            if (!rootCaseLoop)
                rootCaseLoop = pointLoop;

            builder.setInsertionPointToStart(pointLoop.getBody());
            pointIvs.push_back(pointLoop.getInductionVar());
        } else {
            // tail case: 直接在第二阶段生成 untiled tail loop
            auto tailLoop =
                createStaticFor(builder, loc, dim.fullUb, loop.ub, loop.step);
            if (!rootCaseLoop)
                rootCaseLoop = tailLoop;

            builder.setInsertionPointToStart(tailLoop.getBody());
            pointIvs.push_back(tailLoop.getInductionVar());
        }
    }

    return cloneBody(builder, loopInfos, pointIvs, bodyOps);
}

mlir::LogicalResult buildTilingNest(
    std::vector<LoopInfo> loopInfos,
    llvm::ArrayRef<unsigned> tileSizes,
    llvm::SmallVectorImpl<mlir::affine::AffineForOp> *generatedCaseRoots) {
    auto anchor = loopInfos.front().for_op;
    auto loc = anchor.getLoc();

    llvm::SmallVector<mlir::Operation *, 16> bodyOps;
    collectInnermostBodyOps(loopInfos.back().for_op, bodyOps);

    auto dims = computeTileDimInfos(loopInfos, tileSizes);

    mlir::OpBuilder builder(anchor);
    uint64_t caseCount = 1ULL << loopInfos.size();

    llvm::SmallVector<mlir::affine::AffineForOp, 8> roots;
    for (uint64_t mask = 0; mask < caseCount; ++mask) {
        if (!shouldBuildCase(dims, mask))
            continue;

        builder.setInsertionPoint(anchor);

        mlir::affine::AffineForOp rootCaseLoop = nullptr;
        if (mlir::failed(buildOneCase(builder, loc, loopInfos, dims, mask, bodyOps,
                                      rootCaseLoop))) {
            return mlir::failure();
        }

        if (rootCaseLoop) {
            roots.push_back(rootCaseLoop);
        }
    }

    if (generatedCaseRoots) {
        generatedCaseRoots->append(roots.begin(), roots.end());
    }

    anchor.erase();
    return mlir::success();
}

}
