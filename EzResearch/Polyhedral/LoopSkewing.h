//===-- LoopSkewing.h ----------------------------------------- -*- C++ -*-===//
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


#ifndef EZ_RESEARCH_LOOP_SKEWING_H
#define EZ_RESEARCH_LOOP_SKEWING_H

#include "AffineSystem.h"
#include "PolyhedralInfo.h"

namespace ezresearch {

/// 用于求解变换矩阵
Matrix SolveSkewingMatrix(PolyhedralInfo polyhedral_info);

/// 检测是否满足约束
bool checkConstraint(Matrix matrix, std::vector<Dependence> &dependence);

/// 计算转换后最大依赖距离（返回字典序最大的距离向量）
std::vector<int64_t> calculateMaxDependencyDistance(Matrix matrix, std::vector<Dependence> &dependence);

/// 对非均匀依赖求解字典序最小距离向量
std::vector<int64_t> solveLexicographicMinDistance(
    const Matrix& matrix,
    const Dependence& dep);

/// 字典序比较：a > b ?
bool isLexicographicallyGreater(const std::vector<int64_t>& a, const std::vector<int64_t>& b);

}

#endif //EZ_RESEARCH_LOOP_SKEWING_H