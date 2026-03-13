//===-- OptLoopPeeling.cpp ------------------------------------ -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Loop Peeling Pass for Affine For Loops with Vectorized Operations
//
// This pass handles the case where (ub - lb) % step != 0 in affine.for loops
// that contain vectorized operations. It peels the remaining iterations and
// converts them to scalar operations.
//
//===----------------------------------------------------------------------===//


#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Casting.h"

namespace ezresearch {

/// 检查循环体是否包含向量化操作
static bool hasVectorOps(mlir::affine::AffineForOp forOp) {
    bool found = false;
    forOp.getBody()->walk([&](mlir::vector::TransferReadOp) {
        found = true;
        return mlir::WalkResult::interrupt();
    });
    return found;
}

/// 将向量类型转换为对应的标量类型
static mlir::Type getScalarType(mlir::Type type) {
    if (auto vecType = llvm::dyn_cast<mlir::VectorType>(type)) {
        return vecType.getElementType();
    }
    return type;
}

/// 将 affine.apply 转换为 arith 操作，以便更好地进行常量折叠
/// 例如：affine_map<(d0) -> (d0 - 1)>(%x) -> arith.subi %x, %c1
/// 注意：在 affine 表达式中，减法被表示为 Add 加负数
static mlir::Value convertAffineApplyToArith(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::affine::AffineApplyOp applyOp, mlir::IRMapping &mapping) {
    llvm::SmallVector<mlir::Value> operands;
    for (mlir::Value operand : applyOp.getMapOperands()) {
        operands.push_back(mapping.lookupOrDefault(operand));
    }

    return mlir::affine::AffineApplyOp::create(builder, loc, applyOp.getAffineMap(), operands).getResult();
}

/// 将向量常量转换为标量常量
/// 如果 value 是向量常量，提取第一个元素；否则返回原值
static mlir::Value scalarizeVectorConstant(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value value, mlir::IRMapping &mapping) {

    value = mapping.lookupOrDefault(value);
    if (!llvm::isa<mlir::VectorType>(value.getType())) {
        return value;
    }
    return mlir::vector::ExtractOp::create(builder, loc, value, mlir::ArrayRef<int64_t>{0});
}

/// 生成单个剥离迭代的标量化 IR
template <typename Range>
static llvm::SmallVector<mlir::Value> remapValues(Range values, mlir::IRMapping &mapping) {
    llvm::SmallVector<mlir::Value> result;
    for (mlir::Value v : values) {
        result.push_back(mapping.lookupOrDefault(v));
    }
    return result;
}

static void mapResults(mlir::Operation *oldOp, mlir::Operation *newOp, mlir::IRMapping &mapping) {
    for (auto [oldRes, newRes] : llvm::zip(oldOp->getResults(), newOp->getResults())) {
        mapping.map(oldRes, newRes);
    }
}

static void generateScalarizedIteration(
    mlir::OpBuilder &builder, mlir::affine::AffineForOp forOp,
    int64_t peelIndexValue, mlir::IRMapping &mapping) {

    mlir::Location loc = forOp.getLoc();
    mapping.map(forOp.getInductionVar(), mlir::arith::ConstantIndexOp::create(builder, loc, peelIndexValue));

    for (mlir::Operation &op : forOp.getBody()->getOperations()) {
        if (llvm::isa<mlir::affine::AffineYieldOp>(op)) {
            continue;
        }

        if (auto readOp = llvm::dyn_cast<mlir::vector::TransferReadOp>(&op)) {
            auto loadOp = mlir::memref::LoadOp::create(
                builder, loc, readOp.getBase(), remapValues(readOp.getIndices(), mapping));
            mapping.map(readOp.getResult(), loadOp.getResult());
            continue;
        }

        if (auto writeOp = llvm::dyn_cast<mlir::vector::TransferWriteOp>(&op)) {
            mlir::memref::StoreOp::create(
                builder, loc, mapping.lookupOrDefault(writeOp.getVector()),
                writeOp.getBase(), remapValues(writeOp.getIndices(), mapping));
            continue;
        }

        if (auto applyOp = llvm::dyn_cast<mlir::affine::AffineApplyOp>(&op)) {
            mapping.map(applyOp.getResult(), convertAffineApplyToArith(builder, loc, applyOp, mapping));
            continue;
        }

        if (op.getDialect() && op.getDialect()->getNamespace() == "arith") {
            llvm::SmallVector<mlir::Value> operands;
            for (mlir::Value operand : op.getOperands()) {
                operands.push_back(scalarizeVectorConstant(builder, loc, operand, mapping));
            }

            llvm::SmallVector<mlir::Type> resultTypes;
            for (mlir::Type t : op.getResultTypes()) {
                resultTypes.push_back(getScalarType(t));
            }

            mlir::Operation *newOp = builder.create(
                loc, op.getName().getIdentifier(), operands, resultTypes, op.getAttrs());
            mapResults(&op, newOp, mapping);
            continue;
        }

        mlir::Operation *newOp = builder.clone(op, mapping);
        mapResults(&op, newOp, mapping);
    }
}

struct OptLoopPeelingPass : public mlir::PassWrapper<OptLoopPeelingPass, mlir::OperationPass<mlir::ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OptLoopPeelingPass)

    llvm::StringRef getArgument() const final { return "loop-peeling"; }

    void runOnOperation() override {
        mlir::ModuleOp module = getOperation();
        mlir::IRRewriter rewriter(module.getContext());

        // 收集所有需要处理的 affine.for（从内到外）
        llvm::SmallVector<mlir::affine::AffineForOp> loopsToProcess;
        module.walk([&](mlir::affine::AffineForOp forOp) {
            if (!forOp.hasConstantLowerBound() ||
                !forOp.hasConstantUpperBound() ||
                !hasVectorOps(forOp)) {
                return;
            }

            int64_t lb = forOp.getConstantLowerBound();
            int64_t ub = forOp.getConstantUpperBound();
            int64_t step = forOp.getStep().getSExtValue();

            if ((ub - lb) % step != 0) {
                loopsToProcess.push_back(forOp);
            }
        });

        // 处理每个需要 peeling 的循环
        for (mlir::affine::AffineForOp forOp : loopsToProcess) {
            int64_t lb = forOp.getConstantLowerBound();
            int64_t ub = forOp.getConstantUpperBound();
            int64_t step = forOp.getStep().getSExtValue();
            int64_t peelCount = (ub - lb) % step;
            int64_t newUb = ub - peelCount;

            // 1. 调整原循环的上界
            rewriter.setInsertionPoint(forOp);
            rewriter.startOpModification(forOp);
            forOp.setConstantUpperBound(newUb);
            rewriter.finalizeOpModification(forOp);

            // 2. 为每个剥离索引生成标量化 IR
            rewriter.setInsertionPointAfter(forOp);
            for (int64_t i = newUb; i < ub; ++i) {
                mlir::IRMapping mapping;
                generateScalarizedIteration(rewriter, forOp, i, mapping);
            }
        }
    }
};

void registerOptLoopPeelingPass() {
    mlir::PassRegistration<OptLoopPeelingPass>();
}

std::unique_ptr<mlir::Pass> createOptLoopPeelingPass() {
    return std::make_unique<OptLoopPeelingPass>();
}

}