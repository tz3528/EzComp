//===-- BuildTilingNest.h ------------------------------------- -*- C++ -*-===//
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


#ifndef EZ_RESEARCH_BUILD_TILING_NEST_H
#define EZ_RESEARCH_BUILD_TILING_NEST_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"

#include "Utils/QueryUtil.h"

namespace ezresearch {

/// 用显式 full/tail 版本化替代 tilePerfectlyNested，避免 affine.min。
///
/// 约束：
/// 1. loopInfos 必须按 band 外到内顺序给出。
/// 2. band 必须是 perfect nest。
/// 3. 当前实现假定循环已经 normalize 到 step == 1；
///    如果你后续要支持一般 step，需要把 tileSize 解释成“迭代次数”并重写 fullTrip 计算。
/// 4. ub 语义为 [lb, ub)。
///
/// 生成策略：
/// - mask = 0      : 全维 full-tile 主路径
/// - mask != 0     : 显式 tail/full 组合路径，总数最多 2^d - 1
///
/// 成功后会删除原始 band，并在原 band 前面插入新生成的多个 loop nest。
///
/// generatedCaseRoots:
///   可选输出，返回每个生成 case 的最外层 root loop。
mlir::LogicalResult buildTilingNest(
    std::vector<LoopInfo> loopInfos,
    llvm::ArrayRef<unsigned> tileSizes,
    llvm::SmallVectorImpl<mlir::affine::AffineForOp> *generatedCaseRoots = nullptr);

}

#endif //EZ_RESEARCH_BUILD_TILING_NEST_H