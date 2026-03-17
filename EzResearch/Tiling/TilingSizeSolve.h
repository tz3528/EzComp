//===-- TilingSizeSolve.h ------------------------------------- -*- C++ -*-===//
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


#ifndef EZ_RESEARCH_TILING_SIZE_SOLVE_H
#define EZ_RESEARCH_TILING_SIZE_SOLVE_H

#include <vector>

#include "Utils/QueryUtil.h"

namespace ezresearch {

constexpr double alpha = 0.5;

uint64_t computeHalo(std::vector<uint64_t>& tiling_size, uint64_t B);

void solveDBD(
    std::vector<LoopInfo>& loop_infos,
    std::vector<uint64_t>& tiling_size,
    std::vector<uint64_t>& ans_size,
    uint64_t B, uint64_t M, uint64_t& ans_total, uint64_t& ans_halo);

std::vector<uint64_t> TilingSizeSolve(std::vector<LoopInfo>& loop_infos, uint64_t B, uint64_t M);

}

#endif //EZ_RESEARCH_TILING_SIZE_SOLVE_H
