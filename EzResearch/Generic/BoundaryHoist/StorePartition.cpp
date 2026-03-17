//===-- StorePartition.cpp ------------------------------------ -*- C++ -*-===//
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


#include "StorePartition.h"

namespace ezresearch {

void StorePartition::emplace(mlir::memref::StoreOp store) {
    mlir::ValueRange indices = store.getIndices();

    if (auto cstOp = indices[0].getDefiningOp<mlir::arith::ConstantOp>()) {
        if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(cstOp.getValue())) {
            if (intAttr.getInt() == 0) {
                return;
            }
        }
    }

    for (size_t i = 1; i < indices.size(); ++i) {
        if (!isFromAffineForIV(indices[i])) {
            boundary_store.emplace_back(store);
            return ;
        }
    }
    iter_store.emplace_back(store);
}

std::vector<mlir::memref::StoreOp> StorePartition::analyze() {
    std::vector<mlir::memref::StoreOp> result;

    for (auto &boundary : boundary_store) {
        auto bIdx = boundary.getIndices();
        bool canHoist = true;

        for (auto &iter : iter_store) {
            auto iIdx = iter.getIndices();

            // 第0维：时间维，不同则跳过
            if (bIdx[0] != iIdx[0]) {
                continue;
            }

            if (!proveNoSpatialOverlap(boundary, iter)) {
                canHoist = false;
                break;
            }
        }

        if (canHoist) {
            result.push_back(boundary);
        }
    }
    return result;
}

bool StorePartition::proveNoOverlap1D(mlir::Value a, mlir::Value b) {
    auto ca = getConstantIndex(a);
    auto cb = getConstantIndex(b);

    if (ca && cb) {
        return *ca != *cb;
    }

    auto la = getLoopInfoFromAffineIV(a);
    auto lb = getLoopInfoFromAffineIV(b);

    if (ca && lb) {
        return *ca < lb->lb || *ca >= lb->ub;
    }

    if (la && cb) {
        return *cb < la->lb || *cb >= la->ub;
    }

    if (la && lb) {
        return la->ub <= lb->lb || lb->ub <= la->lb;
    }

    return false;
}

bool StorePartition::proveNoSpatialOverlap(mlir::memref::StoreOp boundary, mlir::memref::StoreOp iter) {
    auto bIdx = boundary.getIndices();
    auto iIdx = iter.getIndices();

    // 存在某一维二者不交，则认为这两个sotre操作不交
    for (size_t d = 1; d < bIdx.size(); ++d) {
        if (proveNoOverlap1D(bIdx[d], iIdx[d])) {
            return true;
        }
    }
    return false;
}

}