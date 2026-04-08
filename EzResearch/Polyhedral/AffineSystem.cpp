//===-- AffineSystem.cpp -------------------------------------- -*- C++ -*-===//
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


#include "llvm/ADT/DenseSet.h"

#include <algorithm>
#include <queue>

#include "AffineSystem.h"

namespace ezresearch {

void Matrix::dump() const {
    llvm::errs() << "=== Matrix Dump ===\n";

    if (matrix.empty()) {
        llvm::errs() << "  (Empty Matrix)\n";
        llvm::errs() << "===================\n";
        return;
    }

    for (const auto& [row_idx, affine_info] : matrix) {
        llvm::errs() << "Row " << row_idx << ": ";

        if (!affine_info.is_affine) {
            llvm::errs() << "[Non-Affine] ";
        }

        // 1. 提取所有列索引并排序，保证输出确定性 (Deterministic Output)
        std::vector<uint32_t> col_indices;
        for (const auto& [col_idx, coeff] : affine_info.coefficient) {
            col_indices.push_back(col_idx);
        }
        std::sort(col_indices.begin(), col_indices.end());

        // 2. 格式化打印多项式表达式
        bool is_first_term = true;
        for (uint32_t col_idx : col_indices) {
            // DenseMap 的 lookup 会在找不到时返回 0，但我们遍历的是已有的 key
            int64_t coeff = affine_info.coefficient.lookup(col_idx);
            if (coeff == 0) continue; // 略过系数为 0 的项

            // 符号处理
            if (!is_first_term && coeff > 0) {
                llvm::errs() << " + ";
            } else if (coeff < 0) {
                llvm::errs() << (is_first_term ? "-" : " - ");
            }

            // 系数处理 (如果是 1 或 -1 且带有维度变量，则省略数字 1)
            int64_t abs_coeff = std::abs(coeff);
            if (abs_coeff != 1) {
                llvm::errs() << abs_coeff << "*";
            }

            // 打印维度变量，例如 d0, d1
            llvm::errs() << "d" << col_idx;
            is_first_term = false;
        }

        // 3. 打印常数项 (Constant)
        if (affine_info.constant != 0 || is_first_term) {
            if (!is_first_term && affine_info.constant > 0) {
                llvm::errs() << " + ";
            } else if (!is_first_term && affine_info.constant < 0) {
                llvm::errs() << " - ";
            }

            // 如果前面有项，只需打印常数绝对值；如果是唯一项，带上本来符号
            if (is_first_term) {
                llvm::errs() << affine_info.constant;
            } else {
                llvm::errs() << std::abs(affine_info.constant);
            }
        }

        llvm::errs() << "\n";
    }
    llvm::errs() << "===================\n";
}

z3::expr lex_less(z3::context& ctx, std::vector<z3::expr>& X, std::vector<z3::expr>& Y) {
    int n = (int)X.size();
    std::vector<z3::expr> eq;
    eq.reserve(n + 1);
    eq.push_back(ctx.bool_val(true)); // 前0项相等

    for (int i = 0; i < n; ++i) {
        eq.push_back(eq.back() && (X[i] == Y[i]));
    }

    z3::expr res = ctx.bool_val(false);
    for (int k = 0; k < n; ++k) {
        res = res || (eq[k] && (X[k] < Y[k]));
    }
    return res;
}

Matrix getInversionMatrix(const Matrix& L) {
    std::vector<uint32_t> rows;
    for (auto &[r, _] : L) {
        rows.push_back(r);
    }

    Matrix inv;
    size_t n = rows.size();

    for (size_t j = 0; j < n; ++j) {
        std::vector<int64_t> x(n);
        for (size_t i = 0; i < n; ++i) {
            const auto &row = L.at(rows[i]);
            int64_t rhs = (i == j);
            for (size_t k = 0; k < i; ++k) {
                auto it = row.coefficient.find(rows[k]);
                if (it != row.coefficient.end()) {
                    rhs -= it->second * x[k];
                }
            }
            auto it = row.coefficient.find(rows[i]);
            if (it == row.coefficient.end() || (it->second != 1 && it->second != -1)) {
                return {};
            }
            x[i] = rhs * it->second;
        }
        for (size_t i = 0; i < n; ++i) {
            if (x[i]) {
                inv[rows[i]].coefficient[rows[j]] = x[i];
            }
        }
    }
    return inv;
}

std::vector<int64_t> multiplyMatrixVector(
    const Matrix& matrix,
    const std::vector<int>& vec) {
    
    // 收集并排序所有行索引
    std::vector<uint32_t> rows;
    for (const auto &[row, _] : matrix) {
        rows.push_back(row);
    }
    std::sort(rows.begin(), rows.end());
    
    std::vector<int64_t> result(rows.size(), 0);
    
    for (size_t i = 0; i < rows.size(); ++i) {
        const AffineInfo &row = matrix.at(rows[i]);
        // result[i] = row.constant + Σ(row.coefficient[j] * vec[j])
        result[i] = row.constant;
        for (size_t j = 0; j < vec.size() && j < rows.size(); ++j) {
            auto it = row.coefficient.find(rows[j]);
            if (it != row.coefficient.end()) {
                result[i] += it->second * vec[j];
            }
        }
    }
    
    return result;
}

AffineInfo substituteIndex(
    const AffineInfo& info,
    const Matrix& S) {
    
    AffineInfo result;
    result.constant = info.constant;
    result.is_affine = info.is_affine;
    
    // 对于 info 中的每个变量 var，用 S[var] 替换
    // 即：coe * var 变成 coe * S[var]
    for (const auto &[var, coe] : info.coefficient) {
        auto it = S.find(var);
        if (it == S.end()) {
            // S 中没有这个变量的定义，保持原样（或者报错）
            result.coefficient[var] += coe;
            continue;
        }
        
        const AffineInfo &sub = it->second;
        // coe * S[var] = coe * (sub.constant + Σ sub.coefficient[k] * k)
        result.constant += coe * sub.constant;
        for (const auto &[k, sub_coe] : sub.coefficient) {
            result.coefficient[k] += coe * sub_coe;
        }
    }
    
    result.normalize();
    return result;
}

std::vector<AffineInfo> substituteIndices(
    const std::vector<AffineInfo>& indices,
    const Matrix& S) {
    
    std::vector<AffineInfo> result;
    result.reserve(indices.size());
    
    for (const auto &info : indices) {
        result.push_back(substituteIndex(info, S));
    }
    
    return result;
}

bool isLexicographicallyNonNegative(const std::vector<int64_t>& distance) {
    for (size_t i = 0; i < distance.size(); ++i) {
        if (distance[i] > 0) {
            return true;   // 找到第一个正分量，满足字典序非负
        }
        if (distance[i] < 0) {
            return false;  // 找到第一个负分量，不满足
        }
        // distance[i] == 0，继续检查下一维
    }
    return true;  // 全为零，也算非负
}

bool SolveDependence(
    std::vector<AffineInfo> &src_constraint,
    std::vector<AffineInfo> &dst_constraint,
    std::vector<AffineInfo> &src_indices,
    std::vector<AffineInfo> &dst_indices) {
    
    // 1. 分别收集 src 和 dst 约束中出现的变量
    llvm::DenseSet<uint32_t> src_vars, dst_vars;
    for (const auto &c : src_constraint) {
        for (const auto &[var, _] : c.coefficient) src_vars.insert(var);
    }
    for (const auto &c : dst_constraint) {
        for (const auto &[var, _] : c.coefficient) dst_vars.insert(var);
    }

    z3::context ctx;
    z3::solver solver(ctx);

    std::map<uint32_t, z3::expr> X_map, Y_map;
    std::vector<z3::expr> shared_X, shared_Y;

    // 2. 构造变量并区分共有变量（用于字典序比较）
    std::vector<uint32_t> ordered_src(src_vars.begin(), src_vars.end());
    std::vector<uint32_t> ordered_dst(dst_vars.begin(), dst_vars.end());
    std::sort(ordered_src.begin(), ordered_src.end());
    std::sort(ordered_dst.begin(), ordered_dst.end());

    // 创建 src 的 X 变量
    for (auto var : ordered_src) {
        X_map.emplace(var, ctx.int_const(("x_" + std::to_string(var)).c_str()));
    }
    // 创建 dst 的 Y 变量
    for (auto var : ordered_dst) {
        Y_map.emplace(var, ctx.int_const(("y_" + std::to_string(var)).c_str()));
    }

    // 提取公共循环变量用于字典序比较
    for (auto var : ordered_src) {
        if (dst_vars.contains(var)) {
            shared_X.push_back(X_map.at(var));
            shared_Y.push_back(Y_map.at(var));
        }
    }

    // 3. 添加字典序约束 X < Y（只比较共享的循环层级）
    if (!shared_X.empty()) {
        solver.add(lex_less(ctx, shared_X, shared_Y));
    }

    // 4. 独立施加多面体域约束（支持任意仿射约束）
    for (const auto &c : src_constraint) {
        z3::expr expr_x = ctx.int_val(c.constant);
        for (const auto &[var, coe] : c.coefficient) {
            expr_x = expr_x + ctx.int_val(coe) * X_map.at(var);
        }
        solver.add(expr_x >= 0);
    }

    for (const auto &c : dst_constraint) {
        z3::expr expr_y = ctx.int_val(c.constant);
        for (const auto &[var, coe] : c.coefficient) {
            expr_y = expr_y + ctx.int_val(coe) * Y_map.at(var);
        }
        solver.add(expr_y >= 0);
    }

    // 5. 添加访存等式约束：src[i] == dst[i]
    for (size_t i = 0; i < src_indices.size(); ++i) {
        z3::expr diff = ctx.int_val(src_indices[i].constant - dst_indices[i].constant);
        for (const auto &[var, coe] : src_indices[i].coefficient) {
            diff = diff + ctx.int_val(coe) * X_map.at(var);
        }
        for (const auto &[var, coe] : dst_indices[i].coefficient) {
            diff = diff - ctx.int_val(coe) * Y_map.at(var);
        }
        solver.add(diff == 0);
    }

    // 6. 求解
    return solver.check() == z3::sat;
}

}
