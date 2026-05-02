//===-- PolyhedralInfo.h -------------------------------------- -*- C++ -*-===//
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


#ifndef EZ_RESEARCH_POLYHEDRAL_INFO_H
#define EZ_RESEARCH_POLYHEDRAL_INFO_H

#include "llvm/ADT/DenseMap.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"

#include "AffineSystem.h"

namespace ezresearch {

struct AccessInfo {
    mlir::Operation *op;                                // 这次访问对应的操作
    mlir::Value memref;                                 // 访问的 memref 句柄
    std::vector<AffineInfo> indices;                    // 每一维下标的仿射信息
    bool is_write;                                      // store=true, load=false
    std::vector<AffineInfo> domain;                     // 循环上下文迭代域
};

enum class DependenceKind {
    RAW,    // 写后读
    WAR,    // 读后写
    WAW     // 写后写
};

enum class DepDirection {
    LESS,       // <  (例如 dst 发生在 src 之后的迭代)
    EQUAL,      // == (例如 dst 和 src 发生在同一迭代)
    GREATER,    // >  (例如 dst 发生在 src 之前的迭代，通常意味着反因果律)
    STAR        // * (距离非常数，或跨越整个维度的复杂依赖)
};

struct Dependence {
    mlir::Operation *src;
    mlir::Operation *dst;
    mlir::Value src_memref;
    mlir::Value dst_memref;
    std::vector<AffineInfo> src_indices;        // src各维度访存信息
    std::vector<AffineInfo> dst_indices;        // dst各维度访存信息
    std::vector<AffineInfo> src_domain;         // src迭代域的仿射约束
    std::vector<AffineInfo> dst_domain;         // dst迭代域的仿射约束
    DependenceKind kind;
    size_t shared_depth = 0;                    // 共享的循环嵌套层数 (Shared Depth)
    std::vector<int> distance_vector;           // 依赖距离向量 (Distance Vector)
    std::vector<DepDirection> direction_vector; // 依赖方向向量 (Direction Vector)
    bool is_uniform = true;                     // 常量距离标志 (Is Uniform)
    bool is_reverse  = false;                   // 判读src在语句顺序上是否晚于dst
};

/// 用于判断在每个索引都有且仅有一个左右边界的情况下，
/// 是否存在两组迭代点X<Y，
/// 使得src在X迭代点访问的内存和dst在Y迭代点访问的内存相同
bool SolveDependenceNoEqual(Dependence &candidate);

/// 用于判断在每个索引都有且仅有一个左右边界的情况下，
/// 是否存在两组迭代点X=Y，
/// 使得src在X迭代点访问的内存和dst在Y迭代点访问的内存相同
bool SolveDependenceEqual(Dependence &candidate);

class PolyhedralInfo {
public:
    PolyhedralInfo(mlir::affine::AffineForOp root);

    llvm::DenseMap<mlir::Value, uint32_t> index_to_id;
    llvm::DenseMap<uint32_t, mlir::Value> id_to_index;
    std::vector<Dependence> dependence_polyhedron;
    uint64_t R;                                         // 模板半径

private:
    void analyze(mlir::Operation *op);

    void analyzeAffineFor(mlir::affine::AffineForOp for_op);
    void analyzeContant(mlir::arith::ConstantOp constant_op);
    void analyzeCompute(mlir::Operation *op);
    void analyzeMemrefLoad(mlir::memref::LoadOp load_op);
    void analyzeMemrefStore(mlir::memref::StoreOp store_op);
    void analyzeAffineLoad(mlir::affine::AffineLoadOp load_op);
    void analyzeAffineStore(mlir::affine::AffineStoreOp store_op);

    std::vector<AffineInfo> collectIndicesAffineInfo(mlir::ValueRange indices);

    llvm::DenseMap<mlir::Value, AffineInfo> value_affine_map;           // 值运算中的仿射信息

    llvm::DenseMap<mlir::Value, std::vector<AccessInfo>> memref_reads;  // 当前已经执行的读操作
    llvm::DenseMap<mlir::Value, std::vector<AccessInfo>> memref_writes; // 当前已经执行的写操作

    std::vector<AffineInfo> now_domain;                                 // 单调栈记录循环上下文迭代域
};

inline AffineInfo buildAffineInfoFromExpr(
    mlir::AffineExpr expr,
    mlir::ValueRange operands,
    const llvm::DenseMap<mlir::Value, AffineInfo> &value_affine_map) {

    AffineInfo result;
    result.constant = 0;
    result.is_affine = true;

    if (auto dimExpr = llvm::dyn_cast<mlir::AffineDimExpr>(expr)) {
        mlir::Value v = operands[dimExpr.getPosition()];
        auto it = value_affine_map.find(v);
        if (it != value_affine_map.end()) return it->second;

        result.is_affine = false;
        return result;
    }

    if (auto symbolExpr = llvm::dyn_cast<mlir::AffineSymbolExpr>(expr)) {
        mlir::Value v = operands[symbolExpr.getPosition()];
        auto it = value_affine_map.find(v);
        if (it != value_affine_map.end()) return it->second;

        result.is_affine = false;
        return result;
    }

    if (auto constExpr = llvm::dyn_cast<mlir::AffineConstantExpr>(expr)) {
        result.constant = constExpr.getValue();
        return result;
    }

    if (auto binaryExpr = llvm::dyn_cast<mlir::AffineBinaryOpExpr>(expr)) {
        AffineInfo lhs = buildAffineInfoFromExpr(binaryExpr.getLHS(), operands, value_affine_map);
        AffineInfo rhs = buildAffineInfoFromExpr(binaryExpr.getRHS(), operands, value_affine_map);

        switch (binaryExpr.getKind()) {
        case mlir::AffineExprKind::Add:
            return lhs + rhs;
        case mlir::AffineExprKind::Mul:
            return lhs * rhs;
        default:
            result.is_affine = false;
            return result;
        }
    }

    result.is_affine = false;
    return result;
}

inline std::vector<AffineInfo> collectAffineMapIndices(
    mlir::AffineMap map,
    mlir::ValueRange operands,
    const llvm::DenseMap<mlir::Value, AffineInfo> &value_affine_map) {

    std::vector<AffineInfo> indices;
    indices.reserve(map.getNumResults());

    for (mlir::AffineExpr expr : map.getResults()) {
        indices.push_back(buildAffineInfoFromExpr(expr, operands, value_affine_map));
    }

    return indices;
}

}


#endif //EZ_RESEARCH_POLYHEDRAL_INFO_H
