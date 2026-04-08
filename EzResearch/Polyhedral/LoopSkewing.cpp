//===-- LoopSkewing.cpp --------------------------------------- -*- C++ -*-===//
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


#include "LoopSkewing.h"

#include <z3++.h>

#include <algorithm>
#include <climits>

namespace ezresearch {

Matrix SolveSkewingMatrix(PolyhedralInfo polyhedral_info) {

    /// 这里枚举所有可能的倾斜系数和sum，
    /// 对于某个sum，
    /// 首先做各类约束检测
    /// 然后对于所有的约束，计算最大的约束距离
    /// 如果存在解，那么就找到最大约束距离最小的那组变换矩阵作为结果

    auto n = polyhedral_info.id_to_index.size();
    // 辅助函数：构造单位矩阵
    auto makeIdentity = [&]() -> Matrix {
        Matrix I;
        for (uint32_t i = 0; i < n; ++i) {
            AffineInfo info;
            info.constant = 0;
            info.is_affine = true;
            info.coefficient[i] = 1;
            I[i] = info;
        }
        return I;
    };

    // 初始最优结果：单位矩阵
    Matrix best_transform = makeIdentity();
    std::vector<int64_t> best_max_dist;      // 最大依赖距离向量（字典序越小越好）
    bool has_valid_solution = false;         // 是否已找到可行解

    if (n <= 1) {
        return best_transform;
    }

    int off_diag_cnt = static_cast<int>(n * (n - 1) / 2);
    uint64_t max_sum = static_cast<uint64_t>(n * n);   // 系数绝对值和的上界

    std::vector<int64_t> coeffs(off_diag_cnt, 0);

    // 递归枚举 lambda：pos 当前处理的位置，remaining 剩余可分配的绝对值之和，current_sum 当前系数总和
    std::function<void(int, int)> dfs = [&](int pos, uint64_t remaining) -> void {
        if (pos == off_diag_cnt) {
            if (remaining != 0) return;

            // 构建当前候选变换矩阵（下三角，对角线为1）
            Matrix candidate;
            // 临时二维矩阵，方便填充
            std::vector<std::vector<int64_t>> mat(n, std::vector<int64_t>(n, 0));
            for (size_t i = 0; i < n; ++i) mat[i][i] = 1;
            int idx = 0;
            for (size_t i = 1; i < n; ++i) {
                for (size_t j = 0; j < i; ++j) {
                    mat[i][j] = coeffs[idx++];
                }
            }
            for (size_t i = 0; i < n; ++i) {
                AffineInfo info;
                info.constant = 0;
                info.is_affine = true;
                for (size_t j = 0; j < n; ++j) {
                    int64_t c = mat[i][j];
                    if (c != 0) {
                        info.coefficient[static_cast<uint32_t>(j)] = c;
                    }
                }
                candidate[static_cast<uint32_t>(i)] = info;
            }

            // =============================================
            // 检查约束并计算最大依赖距离
            // =============================================
            if (!checkConstraint(candidate, polyhedral_info.dependence_polyhedron)) {
                return;
            }

            // 计算该变换下的最大依赖距离向量
            std::vector<int64_t> max_dist = calculateMaxDependencyDistance(
                candidate, polyhedral_info.dependence_polyhedron);

            // 如果是第一个可行解，或字典序更优，则更新
            if (!has_valid_solution || isLexicographicallyGreater(best_max_dist, max_dist)) {
                best_transform = candidate;
                best_max_dist = max_dist;
                has_valid_solution = true;
            }

            return;
        }

        // 枚举当前非对角线位置的系数（绝对值从0到remaining）
        for (uint64_t v = 0; v <= remaining && v <= polyhedral_info.R; ++v) {
            if (v == 0) {
                coeffs[pos] = 0;
                dfs(pos + 1, remaining);
            } else {
                coeffs[pos] = v;
                dfs(pos + 1, remaining - v);
                coeffs[pos] = -v;
                dfs(pos + 1, remaining - v);
            }
        }
    };

    // 按系数和 sum 从 0 到 max_sum 依次枚举
    for (int sum = 1; sum <= max_sum; ++sum) {
        dfs(0, sum);
        if (has_valid_solution) {
            break;
        }
    }

    return best_transform;
}

bool checkConstraint(Matrix matrix, std::vector<Dependence> &dependence) {
    // 1. 求逆矩阵 S = T⁻¹
    auto S = getInversionMatrix(matrix);
    if (S.empty()) return false;  // 矩阵不可逆，变换非法

    // 2. 对每个依赖检查
    for (const auto &dep : dependence) {
        if (dep.is_uniform) {
            // ===== 均匀依赖 =====

            // 2a. 计算新距离向量 d' = T × distance_vector
            auto new_distance = multiplyMatrixVector(matrix, dep.distance_vector);

            // 2b. 检查依赖方向保持

            // 2c. 检查可分块性：d' 字典序非负
            if (!isLexicographicallyNonNegative(new_distance)) {
                return false;
            }

        } else {
            // ===== 非均匀依赖 =====

            // 2d. 用 S 替换访存索引
            auto new_src_indices = substituteIndices(dep.src_indices, S);
            auto new_dst_indices = substituteIndices(dep.dst_indices, S);

            // 2e. 用 S 替换边界约束
            auto new_src_constraint = substituteIndices(dep.src_domain, S);
            auto new_dst_constraint = substituteIndices(dep.dst_domain, S);

            // 2f. 在新索引空间求解依赖是否存在
            bool has_dependence = SolveDependence(
                new_src_constraint, new_dst_constraint,
                new_src_indices, new_dst_indices);

            // 关键：如果原始存在依赖，变换后必须仍存在
            // 依赖消失会导致执行顺序改变，破坏程序语义
            if (!has_dependence) {
                return false;
            }
        }
    }

    return true;
}

bool isLexicographicallyGreater(const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
    size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        if (a[i] > b[i]) return true;
        if (a[i] < b[i]) return false;
    }
    // 前缀相等，较长向量在字典序上更大（如果它有额外的正分量）
    if (a.size() > b.size()) {
        for (size_t i = n; i < a.size(); ++i) {
            if (a[i] > 0) return true;
            if (a[i] < 0) return false;
        }
    }
    return false;  // 相等
}

std::vector<int64_t> solveLexicographicMinDistance(
    const Matrix& matrix,
    const Dependence& dep) {

    size_t n = dep.shared_depth;
    std::vector<int64_t> min_distance(n, 0);

    if (n == 0) return min_distance;

    // 1. 计算逆矩阵
    auto S = getInversionMatrix(matrix);
    if (S.empty()) {
        // 矩阵不可逆，返回极大值
        return std::vector<int64_t>(n, INT64_MAX);
    }

    // 2. 替换索引和域约束
    auto new_src_indices = substituteIndices(dep.src_indices, S);
    auto new_dst_indices = substituteIndices(dep.dst_indices, S);
    auto new_src_domain = substituteIndices(dep.src_domain, S);
    auto new_dst_domain = substituteIndices(dep.dst_domain, S);

    // 3. 收集变量
    llvm::DenseSet<uint32_t> src_vars, dst_vars;
    for (const auto& c : new_src_domain) {
        for (const auto& [var, coe] : c.coefficient) {
            src_vars.insert(var);
        }
    }
    for (const auto& c : new_dst_domain) {
        for (const auto& [var, coe] : c.coefficient) {
            dst_vars.insert(var);
        }
    }

    // 4. 排序变量（保证外层到内层的顺序）
    std::vector<uint32_t> ordered_src(src_vars.begin(), src_vars.end());
    std::vector<uint32_t> ordered_dst(dst_vars.begin(), dst_vars.end());
    std::sort(ordered_src.begin(), ordered_src.end());
    std::sort(ordered_dst.begin(), ordered_dst.end());

    // 5. 提取共享变量（用于计算距离）
    std::vector<uint32_t> shared_vars;
    for (uint32_t var : ordered_src) {
        if (dst_vars.contains(var)) {
            shared_vars.push_back(var);
        }
    }

    // 6. 构建 Z3 上下文和变量
    z3::context ctx;
    std::map<uint32_t, z3::expr> X_map, Y_map;
    std::vector<z3::expr> X_vec, Y_vec;

    for (uint32_t var : ordered_src) {
        X_map.emplace(var, ctx.int_const(("x_" + std::to_string(var)).c_str()));
    }
    for (uint32_t var : ordered_dst) {
        Y_map.emplace(var, ctx.int_const(("y_" + std::to_string(var)).c_str()));
    }
    for (uint32_t var : shared_vars) {
        X_vec.push_back(X_map.at(var));
        Y_vec.push_back(Y_map.at(var));
    }

    // 7. 构建优化器并添加约束
    z3::optimize opt(ctx);

    // 7a. 添加源迭代域约束
    for (const auto& c : new_src_domain) {
        z3::expr expr = ctx.int_val(c.constant);
        for (const auto& [var, coe] : c.coefficient) {
            expr = expr + ctx.int_val(coe) * X_map.at(var);
        }
        opt.add(expr >= 0);
    }

    // 7b. 添加目标迭代域约束
    for (const auto& c : new_dst_domain) {
        z3::expr expr = ctx.int_val(c.constant);
        for (const auto& [var, coe] : c.coefficient) {
            expr = expr + ctx.int_val(coe) * Y_map.at(var);
        }
        opt.add(expr >= 0);
    }

    // 7c. 添加访存等式约束
    for (size_t i = 0; i < new_src_indices.size() && i < new_dst_indices.size(); ++i) {
        z3::expr diff = ctx.int_val(new_src_indices[i].constant - new_dst_indices[i].constant);
        for (const auto& [var, coe] : new_src_indices[i].coefficient) {
            diff = diff + ctx.int_val(coe) * X_map.at(var);
        }
        for (const auto& [var, coe] : new_dst_indices[i].coefficient) {
            diff = diff - ctx.int_val(coe) * Y_map.at(var);
        }
        opt.add(diff == 0);
    }

    // 7d. 添加字典序约束 X < Y
    if (!X_vec.empty()) {
        opt.add(lex_less(ctx, X_vec, Y_vec));
    }

    // 8. 逐维最小化
    for (size_t i = 0; i < n && i < X_vec.size(); ++i) {
        z3::expr dist_i = Y_vec[i] - X_vec[i];
        opt.minimize(dist_i);

        if (opt.check() == z3::sat) {
            z3::model m = opt.get_model();
            min_distance[i] = m.eval(dist_i).get_numeral_int64();
            // 固定当前维度
            opt.add(dist_i == ctx.int_val(min_distance[i]));
        } else {
            // 不可满足，说明约束有误
            min_distance[i] = INT64_MAX;
        }
    }

    return min_distance;
}

std::vector<int64_t> calculateMaxDependencyDistance(
    Matrix matrix,
    std::vector<Dependence>& dependence) {

    std::vector<int64_t> max_distance_vector;

    for (const auto& dep : dependence) {
        std::vector<int64_t> dist;

        if (dep.is_uniform) {
            // 均匀依赖：直接计算变换后的距离向量
            dist = multiplyMatrixVector(matrix, dep.distance_vector);
        } else {
            // 非均匀依赖：求解字典序最小距离
            dist = solveLexicographicMinDistance(matrix, dep);
        }

        // 更新字典序最大值
        if (isLexicographicallyGreater(dist, max_distance_vector)) {
            max_distance_vector = dist;
        }
    }

    return max_distance_vector;
}

}
