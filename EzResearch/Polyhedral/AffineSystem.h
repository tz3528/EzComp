//===-- AffineSystem.h ---------------------------------------- -*- C++ -*-===//
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


#ifndef EZ_RESEARCH_AFFINE_SYSEM_H
#define EZ_RESEARCH_AFFINE_SYSEM_H

#include "mlir/Dialect/Affine/IR/AffineOps.h"

#include <z3++.h>

namespace ezresearch {

/// 仿射信息，用于记录每个索引的系数及仿射偏移量
/// 当表示为线性约束时，不等号方向为 >= 0
/// 这里存在一个假设，所有循环的步长都是1，步长非1的由索引部分解决
struct  AffineInfo {
    int64_t constant = 0;
    llvm::DenseMap<uint32_t, int64_t> coefficient;
    bool is_affine = true;

    bool isConstant() const {
        return is_affine && coefficient.empty();
    }

    void normalize() {
        for (auto it = coefficient.begin(); it != coefficient.end();) {
            if (it->second == 0)
                coefficient.erase(it++);
            else
                ++it;
        }
    }

    AffineInfo operator+(const AffineInfo &other) const {
        AffineInfo result;
        result.constant = constant + other.constant;
        result.coefficient = coefficient;

        for (const auto &[op, coe] : other.coefficient) {
            result.coefficient[op] += coe;
        }

        result.is_affine = is_affine && other.is_affine;
        result.normalize();
        return result;
    }

    AffineInfo operator-(const AffineInfo &other) const {
        AffineInfo result;
        result.constant = constant - other.constant;
        result.coefficient = coefficient;

        for (const auto &[op, coe] : other.coefficient) {
            result.coefficient[op] -= coe;
        }

        result.is_affine = is_affine && other.is_affine;
        result.normalize();
        return result;
    }

    AffineInfo operator*(const AffineInfo &other) const {
        AffineInfo result;

        // 只要有一边已经不是仿射，结果直接不是仿射
        if (!is_affine || !other.is_affine) {
            result.is_affine = false;
            return result;
        }

        // 两边都含变量 => 非仿射
        if (!coefficient.empty() && !other.coefficient.empty()) {
            result.is_affine = false;
            return result;
        }

        result.constant = constant * other.constant;
        if (coefficient.empty() && other.coefficient.empty()) {
            return result;
        }

        if (!coefficient.empty() && other.coefficient.empty()) {
            for (const auto &[op, coe] : coefficient) {
                result.coefficient[op] = coe * other.constant;
            }
            result.normalize();
            return result;
        }

        if (coefficient.empty() && !other.coefficient.empty()) {
            for (const auto &[op, coe] : other.coefficient) {
                result.coefficient[op] = coe * constant;
            }
            result.normalize();
            return result;
        }

        return result;
    }

    AffineInfo operator/(const AffineInfo &other) const {
        AffineInfo result;

        // 除数为 0
        if (other.isConstant() && other.constant == 0) {
            result.is_affine = false;
            return result;
        }

        // 只要有一边已经不是仿射，结果直接不是仿射
        if (!is_affine || !other.is_affine) {
            result.is_affine = false;
            return result;
        }

        // 仿射表达式 / 含变量表达式 => 非仿射
        if (!other.coefficient.empty()) {
            result.is_affine = false;
            return result;
        }

        // 现在 other 一定是纯常数
        int64_t divisor = other.constant;

        // 系数必须能整除
        if (constant % divisor != 0) {
            result.is_affine = false;
            return result;
        }

        for (const auto &[op, coe] : coefficient) {
            if (coe % divisor != 0) {
                result.is_affine = false;
                return result;
            }
        }

        result.constant = constant / divisor;
        for (const auto &[op, coe] : coefficient) {
            result.coefficient[op] = coe / divisor;
        }
        result.is_affine = true;
        result.normalize();
        return result;
    }

    bool operator==(const AffineInfo &other) const {
        if (is_affine != other.is_affine) return false;
        if (constant != other.constant) return false;
        return coefficient == other.coefficient;
    }

    bool operator!=(const AffineInfo &other) const {
        return !(*this == other);
    }
};

struct Matrix {
    std::map<uint32_t, AffineInfo> matrix;

    AffineInfo& operator[](uint32_t row_idx) {
        return matrix[row_idx];
    }

    auto begin() {
        return matrix.begin();
    }

    auto end() {
        return matrix.end();
    }

    auto begin() const {
        return matrix.begin();
    }

    auto end()   const {
        return matrix.end();
    }

    bool empty() const {
        return matrix.empty();
    }

    auto find(uint32_t row_idx) {
        return matrix.find(row_idx);
    }

    auto find(uint32_t row_idx) const {
        return matrix.find(row_idx);
    }

    AffineInfo& at(uint32_t row_idx) {
        return matrix.at(row_idx);
    }

    const AffineInfo& at(uint32_t row_idx) const {
        return matrix.at(row_idx);
    }

    void dump() const;
};

// 添加字典序限制，其中要求X<Y
z3::expr lex_less(z3::context& ctx, std::vector<z3::expr>& X, std::vector<z3::expr>& Y);

/// 求解逆矩阵
/// 目前保证输入的变换矩阵，是下三角幺模矩阵
Matrix getInversionMatrix(const Matrix& matrix);

/// 矩阵-向量乘法
/// 计算 matrix × vec，返回新向量
std::vector<int64_t> multiplyMatrixVector(
    const Matrix& matrix,
    const std::vector<int>& vec);

/// 索引替换：将 info 中的每个索引变量用 S 的对应行替换
AffineInfo substituteIndex(
    const AffineInfo& info,
    const Matrix& S);

/// 批量索引替换
std::vector<AffineInfo> substituteIndices(
    const std::vector<AffineInfo>& indices,
    const Matrix& S);

/// 检查距离向量是否字典序非负
/// 即 d ≥ 0（字典序意义）
bool isLexicographicallyNonNegative(const std::vector<int64_t>& distance);

/// 求解依赖是否存在
/// 在 src_constraint 和 dst_constraint 的约束下，判断是否存在 src 和 dst 访问同一内存的解
/// 要求存在 X < Y（字典序）使得 src(X) == dst(Y)
/// 返回 true 表示存在依赖，false 表示不存在依赖
bool SolveDependence(
    std::vector<AffineInfo> &src_constraint,
    std::vector<AffineInfo> &dst_constraint,
    std::vector<AffineInfo> &src_indices,
    std::vector<AffineInfo> &dst_indices);

}


#endif //EZ_RESEARCH_AFFINE_SYSEM_H
