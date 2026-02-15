//===-- CompOps.cpp --------------------------------------------*- C++ -*-===//
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

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "CompOps.h"

#define GET_OP_CLASSES
#include "CompOps.cpp.inc"

namespace ezcompile::comp {

mlir::LogicalResult ApplyInitOp::verify() {
    // 检查 rhs region 是否为空
    mlir::Region &rhs = getRhs();
    if (rhs.empty()) {
        return emitOpError("rhs region must not be empty");
    }

    // 检查 region 的最后一个操作是否是 comp.yield
    mlir::Block &block = rhs.front();
    mlir::Operation *terminator = block.getTerminator();
    if (!terminator) {
        return emitOpError("rhs region must end with a terminator");
    }

    auto yieldOp = mlir::dyn_cast<YieldOp>(terminator);
    if (!yieldOp) {
        return emitOpError("rhs region must end with comp.yield");
    }

    // 检查 yield 是否恰好产出 1 个值
    if (yieldOp.getOperands().size() != 1) {
        return emitOpError("comp.yield must produce exactly 1 value, got ")
            << yieldOp.getOperands().size();
    }

    return mlir::success();
}

mlir::LogicalResult DimOp::verify() {
    auto lower = getLower();
    auto upper = getUpper();

    if (upper <= lower) {
        return emitOpError("upper must be greater than lower");
    }

    auto points = getPoints();
    if (points < 2) {
        return emitOpError("points must be at least 2");
    }

    return mlir::success();
}

mlir::LogicalResult DirichletOp::verify() {
    // 检查 rhs region 是否为空
    mlir::Region &rhs = getRhs();
    if (rhs.empty()) {
        return emitOpError("rhs region must not be empty");
    }

    // 检查 region 的最后一个操作是否是 comp.yield
    mlir::Block &block = rhs.front();
    mlir::Operation *terminator = block.getTerminator();
    if (!terminator) {
        return emitOpError("rhs region must end with a terminator");
    }

    auto yieldOp = mlir::dyn_cast<YieldOp>(terminator);
    if (!yieldOp) {
        return emitOpError("rhs region must end with comp.yield");
    }

    // 检查 yield 是否恰好产出 1 个值
    if (yieldOp.getOperands().size() != 1) {
        return emitOpError("comp.yield must produce exactly 1 value, got ")
            << yieldOp.getOperands().size();
    }

    return mlir::success();
}

mlir::LogicalResult FieldOp::verify() {
    // 获取 ProblemOp（父操作）
    auto *parentOp = (*this)->getParentOp();
    auto problemOp = mlir::dyn_cast<ProblemOp>(parentOp);
    if (!problemOp) {
        return emitOpError("comp.field must be inside comp.problem");
    }

    // 检查 timeDim 是否存在于 ProblemOp 中
    auto timeDim = getTimeDim();
    if (!problemOp.lookupSymbol(timeDim)) {
        return emitOpError("timeDim '") << timeDim.str() << "' not declared in problem";
    }

    // 检查 spaceDims 不包含 timeDim
    auto spaceDims = getSpaceDims();
    for (auto attr : spaceDims) {
        auto dimRef = mlir::dyn_cast<mlir::FlatSymbolRefAttr>(attr);
        if (!dimRef) {
            return emitOpError("spaceDims must contain symbol references");
        }
        if (dimRef.getValue() == timeDim.str()) {
            return emitOpError("timeDim cannot appear in spaceDims");
        }
    }

    // 检查所有 dims 都已声明
    for (auto attr : spaceDims) {
        auto dimRef = mlir::dyn_cast<mlir::FlatSymbolRefAttr>(attr);
        if (!problemOp.lookupSymbol(dimRef)) {
            return emitOpError("spaceDim '") << dimRef.getValue() << "' not declared in problem";
        }
    }

    return mlir::success();
}

mlir::LogicalResult ForTimeOp::verify() {
    // 检查循环参数：lb < ub, step > 0
    auto lb = getLb();
    auto ub = getUb();
    auto step = getStep();

    // 检查 step 是否为常数且大于 0
    auto stepConst = mlir::dyn_cast_or_null<mlir::arith::ConstantOp>(step.getDefiningOp());
    if (!stepConst) {
        return emitOpError("step must be a constant");
    }

    auto stepAttr = mlir::dyn_cast<mlir::IntegerAttr>(stepConst.getValue());
    if (!stepAttr || stepAttr.getInt() <= 0) {
        return emitOpError("step must be a positive constant");
    }

    // 检查 body 是否有且仅有 1 个 block 参数（归纳变量）
    mlir::Region &body = getBody();
    if (body.empty()) {
        return emitOpError("body region must not be empty");
    }

    mlir::Block &block = body.front();
    if (block.getNumArguments() != 1) {
        return emitOpError("body must have exactly 1 induction variable, got ")
            << block.getNumArguments();
    }

    // 检查归纳变量类型是否为 index
    if (!llvm::isa<mlir::IndexType>(block.getArgument(0).getType())) {
        return emitOpError("induction variable must be of index type");
    }

    return mlir::success();
}

mlir::LogicalResult SampleOp::verify() {
    // 检查尺寸约束：indices == dims == shift
    auto indices = getIndices();
    auto dims = getDims();
    auto shift = getShift();

    size_t indicesSize = indices.size();
    size_t dimsSize = dims.size();
    size_t shiftSize = shift.size();

    if (indicesSize != dimsSize) {
        return emitOpError("number of indices (") << indicesSize
            << ") must match number of dims (" << dimsSize << ")";
    }

    if (indicesSize != shiftSize) {
        return emitOpError("number of indices (") << indicesSize
            << ") must match size of shift (" << shiftSize << ")";
    }

    // 检查所有 index 参数类型是否为 index
    for (auto idx : indices) {
        if (!llvm::isa<mlir::IndexType>(idx.getType())) {
            return emitOpError("all indices must be of index type");
        }
    }

    return mlir::success();
}

mlir::LogicalResult UpdateOp::verify() {
    // 检查 body region 是否为空
    mlir::Region &body = getBody();
    if (body.empty()) {
        return emitOpError("body region must not be empty");
    }

    // 检查 region 的最后一个操作是否是 comp.yield
    mlir::Block &block = body.front();
    mlir::Operation *terminator = block.getTerminator();
    if (!terminator) {
        return emitOpError("body region must end with a terminator");
    }

    auto yieldOp = mlir::dyn_cast<YieldOp>(terminator);
    if (!yieldOp) {
        return emitOpError("body region must end with comp.yield");
    }

    // 检查 yield 是否恰好产出 1 个值
    if (yieldOp.getOperands().size() != 1) {
        return emitOpError("comp.yield must produce exactly 1 value, got ")
            << yieldOp.getOperands().size();
    }

    return mlir::success();
}

mlir::LogicalResult CallOp::verify() {
    // 检查结果类型是否为 f64
    auto resultType = getResult().getType();
    if (!llvm::isa<mlir::Float64Type>(resultType)) {
        return emitOpError("result must be of f64 type");
    }

    return mlir::success();
}

}