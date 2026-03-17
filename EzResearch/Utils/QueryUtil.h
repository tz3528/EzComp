//===-- QueryUtil.h ------------------------------------------- -*- C++ -*-===//
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


#ifndef EZ_RESEARCH_QUERY_UTIL_H
#define EZ_RESEARCH_QUERY_UTIL_H

#include "llvm/ADT/SmallPtrSet.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Casting.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"

namespace ezresearch {

inline bool isFromAffineForIV(mlir::Value v, SmallPtrSetImpl<mlir::Value> &visited) {
    if (!visited.insert(v).second)
        return false;

    // 1) 直接是 affine.for 的 induction variable
    if (auto blockArg = llvm::dyn_cast<mlir::BlockArgument>(v)) {
        mlir::Operation *parentOp = blockArg.getOwner()->getParentOp();
        if (auto forOp = llvm::dyn_cast<mlir::affine::AffineForOp>(parentOp)) {
            if (blockArg == forOp.getInductionVar()) {
                return true;
            }
        }
        return false;
    }

    mlir::Operation *def = v.getDefiningOp();
    if (!def) {
        return false;
    }

    // 2) affine.apply 的结果，递归检查其输入
    if (auto applyOp = llvm::dyn_cast<mlir::affine::AffineApplyOp>(def)) {
        for (mlir::Value operand : applyOp.getMapOperands()) {
            if (isFromAffineForIV(operand, visited)) {
                return true;
            }
        }
        return false;
    }

    // 3) arith 中常见的“转型/转换”类操作，继续追溯源操作数
    if (auto castOp = llvm::dyn_cast<mlir::arith::IndexCastOp>(def)) {
        return isFromAffineForIV(castOp.getIn(), visited);
    }

    if (auto castOp = llvm::dyn_cast<mlir::arith::IndexCastUIOp>(def)) {
        return isFromAffineForIV(castOp.getIn(), visited);
    }

    if (auto extOp = llvm::dyn_cast<mlir::arith::ExtSIOp>(def)) {
        return isFromAffineForIV(extOp.getIn(), visited);
    }

    if (auto extOp = llvm::dyn_cast<mlir::arith::ExtUIOp>(def)) {
        return isFromAffineForIV(extOp.getIn(), visited);
    }

    return false;
}

inline bool isFromAffineForIV(mlir::Value v) {
    llvm::SmallPtrSet<mlir::Value, 8> visited;
    return isFromAffineForIV(v, visited);
}

struct LoopInfo {
    int64_t lb;
    int64_t ub;   // [lb, ub)
    int64_t step;
    mlir::affine::AffineForOp for_op;

    bool operator == (const LoopInfo &other) const {
        return for_op == other.for_op;
    }

    bool operator < (const LoopInfo &other) const {
        return for_op < other.for_op;
    }
};

inline void collectAffineForIVs(
    mlir::Value v, SmallPtrSetImpl<mlir::Value> &visitedValues,
    SmallPtrSetImpl<mlir::Value> &ivs) {

    if (!visitedValues.insert(v).second) {
        return;
    }

    // 1) 直接是 affine.for 的 induction variable
    if (auto barg = llvm::dyn_cast<mlir::BlockArgument>(v)) {
        mlir::Operation *parentOp = barg.getOwner()->getParentOp();
        if (auto forOp = llvm::dyn_cast_or_null<mlir::affine::AffineForOp>(parentOp)) {
            if (forOp.getInductionVar() == v) {
                ivs.insert(v);
            }
        }
        return;
    }

    mlir::Operation *def = v.getDefiningOp();
    if (!def) {
        return;
    }

    // 2) 常量，直接停止
    if (mlir::matchPattern(v, mlir::m_Constant())) {
        return;
    }

    // 3) affine.apply：追它的 map operands
    if (auto applyOp = llvm::dyn_cast<mlir::affine::AffineApplyOp>(def)) {
        for (mlir::Value operand : applyOp.getMapOperands()) {
            collectAffineForIVs(operand, visitedValues, ivs);
        }
        return;
    }

    // 4) unrealized_conversion_cast
    if (auto castOp = llvm::dyn_cast<mlir::UnrealizedConversionCastOp>(def)) {
        for (mlir::Value operand : castOp.getOperands()) {
            collectAffineForIVs(operand, visitedValues, ivs);
        }
        return;
    }

    // 5) 对常见 arith 操作，统一追所有 operands
    if (def->getDialect() &&
        def->getDialect()->getNamespace() == mlir::arith::ArithDialect::getDialectNamespace()) {
        for (mlir::Value operand : def->getOperands()) {
            collectAffineForIVs(operand, visitedValues, ivs);
        }
    }
}

inline std::optional<mlir::Value> findUniqueAffineForIVInChain(mlir::Value v) {
    llvm::SmallPtrSet<mlir::Value, 32> visitedValues;
    llvm::SmallPtrSet<mlir::Value, 4> ivs;
    collectAffineForIVs(v, visitedValues, ivs);
    if (ivs.size() != 1) {
        return std::nullopt;
    }
    return *ivs.begin();
}

inline std::optional<LoopInfo> getLoopInfoFromAffineIV(mlir::Value idx) {
    auto iv = findUniqueAffineForIVInChain(idx);
    if (!iv) {
        return std::nullopt;
    }

    auto barg = llvm::dyn_cast<mlir::BlockArgument>(*iv);
    if (!barg) {
        return std::nullopt;
    }

    mlir::Operation *parentOp = barg.getOwner()->getParentOp();
    if (!parentOp) {
        return std::nullopt;
    }

    auto forOp = llvm::dyn_cast<mlir::affine::AffineForOp>(parentOp);
    if (!forOp) {
        return std::nullopt;
    }

    if (forOp.getInductionVar() != *iv) {
        return std::nullopt;
    }

    if (!forOp.hasConstantLowerBound() || !forOp.hasConstantUpperBound()) {
        return std::nullopt;
    }

    return LoopInfo{
        forOp.getConstantLowerBound(),
        forOp.getConstantUpperBound(),
        forOp.getStep().getSExtValue(),
        forOp
    };
}

inline std::optional<int64_t> getConstantIndex(mlir::Value idx) {
    if (auto cst = idx.getDefiningOp<mlir::arith::ConstantOp>()) {
        if (auto attr = llvm::dyn_cast<mlir::IntegerAttr>(cst.getValue())) {
            return attr.getInt();
        }
    }
    return std::nullopt;
}

}

#endif //EZ_RESEARCH_QUERY_UTIL_H
