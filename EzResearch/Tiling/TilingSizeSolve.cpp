//===-- TilingSizeSolve.cpp ----------------------------------- -*- C++ -*-===//
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


#include "TilingSizeSolve.h"

namespace ezresearch {

uint64_t computeHalo(std::vector<uint64_t>& tiling_size, uint64_t B) {
    uint64_t halo = 0;
    for (size_t i = 0; i < tiling_size.size() - 1; i++) {
        halo += tiling_size[i];
    }
    halo += 2 * tiling_size.back() / B;
    return halo;
}

void solveDBD(
    std::vector<LoopInfo>& loop_infos,
    std::vector<uint64_t>& tiling_size,
    std::vector<uint64_t>& ans_size,
    uint64_t B, uint64_t M, uint64_t& ans_total, uint64_t& ans_halo) {

    if (tiling_size.size() == loop_infos.size()) {
        reverse(tiling_size.begin(), tiling_size.end());
        auto halo = computeHalo(tiling_size, B);
        uint64_t ans = 1, sum = 1;
        for (size_t i = 0; i < loop_infos.size(); i++) {
            ans *= tiling_size[i];
            sum *= (loop_infos[i].ub - loop_infos[i].lb - 1) / tiling_size[i] + 1;
        }
        ans /= B;
        ans += halo;

        if (ans > alpha * M / B) {
            reverse(tiling_size.begin(), tiling_size.end());
            return ;
        }

        ans *= sum;
        if (ans < ans_total) {
            ans_total = ans;
            ans_halo = halo;
            for (size_t i = 0; i < loop_infos.size(); i++) {
                ans_size[i] = tiling_size[i];
            }
        }
        else if (halo < ans_halo) {
            ans_total = ans;
            ans_halo = halo;
            for (size_t i = 0; i < loop_infos.size(); i++) {
                ans_size[i] = tiling_size[i];
            }
        }
        reverse(tiling_size.begin(), tiling_size.end());
        return ;
    }

    auto index = tiling_size.size();
    auto n = loop_infos[index].ub - loop_infos[index].lb + 1;
    if (index == loop_infos.size() - 1) {
        for (auto i = B; i <= n; i += B) {
            tiling_size.push_back(i);
            solveDBD(loop_infos, tiling_size, ans_size, B, M, ans_total, ans_halo);
            tiling_size.pop_back();
        }
    }
    else {
        for (int l = 1, r = 0; l <= n; l = r + 1) {
            int v = (n + l - 1) / l;
            r = (v == 1 ? n : (n - 1) / (v - 1));
            tiling_size.push_back(l);
            solveDBD(loop_infos, tiling_size, ans_size, B, M, ans_total, ans_halo);
            tiling_size.pop_back();
        }
    }
}

std::vector<uint64_t> TilingSizeSolve(std::vector<LoopInfo>& loop_infos, uint64_t B, uint64_t M) {
    std::vector<uint64_t> tiling_size;
    std::vector<uint64_t> ans(loop_infos.size());
    uint64_t total = 0xffffffff, halo = 0xffffffff;
    solveDBD(loop_infos, tiling_size, ans, B, M, total, halo);
    return ans;
}

}
