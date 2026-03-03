//===-- HDF5.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// HDF5 输出接口
// 函数签名匹配 MLIR memref 展开后的字段顺序
//
//===----------------------------------------------------------------------===//


#ifndef EZ_COMPUTE_IO_HDF5_H
#define EZ_COMPUTE_IO_HDF5_H

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// 说明：
// - 函数参数对应 MLIR memref 展开后的字段：
//   (basePtr, data, offset, sizes[0..rank-1], strides[0..rank-1], timeIndex, dimNames, lowers, uppers)
// - Rank 是 memref 的维数：第 0 维是 time-layer（例如 2），后面 (Rank-1) 维是空间维。
// - timeIndex 是时间下标，内部用 layer = timeIndex % sizes[0] 选择要输出的那层。
// - dimNames/lowers/uppers 只描述"空间维"(Rank-1 个)。
// - 输出：当前工作目录下的 result.h5。
//
// 返回：0 成功；负数失败。

int dump_result_hdf5_f64_rank2(
    double* basePtr, double* data, int64_t offset,
    int64_t size0, int64_t size1,
    int64_t stride0, int64_t stride1,
    int64_t timeIndex,
    const char* const* dimNames,
    const double* lowers,
    const double* uppers);

int dump_result_hdf5_f64_rank3(
    double* basePtr, double* data, int64_t offset,
    int64_t size0, int64_t size1, int64_t size2,
    int64_t stride0, int64_t stride1, int64_t stride2,
    int64_t timeIndex,
    const char* const* dimNames,
    const double* lowers,
    const double* uppers);

int dump_result_hdf5_f64_rank4(
    double* basePtr, double* data, int64_t offset,
    int64_t size0, int64_t size1, int64_t size2, int64_t size3,
    int64_t stride0, int64_t stride1, int64_t stride2, int64_t stride3,
    int64_t timeIndex,
    const char* const* dimNames,
    const double* lowers,
    const double* uppers);

#ifdef __cplusplus
}
#endif

#endif // EZ_COMPUTE_IO_HDF5_H