//===-- HDF5Wrapper.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// HDF5 输出接口
// 提供 C ABI 接口，用于将 MLIR memref 数据输出到 HDF5 文件
//
//===----------------------------------------------------------------------===//


#ifndef EZ_COMPUTE_IO_HDF5_WRAPPER_H
#define EZ_COMPUTE_IO_HDF5_WRAPPER_H

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// f64 memref HDF5 输出接口
//===----------------------------------------------------------------------===//

/// 将 f64 2D memref 数据输出到 HDF5 文件
/// @param basePtr   memref 内存基址（未使用）
/// @param data      实际数据指针
/// @param offset    memref 偏移量
/// @param size0/1   memref 各维大小（size0 为时间层数，size1 为空间维）
/// @param stride0/1 memref 各维步长
/// @param timeIndex 时间下标，映射到 layer = timeIndex % size0
/// @param dimNames  空间维度名称数组（rank-1 个）
/// @param lowers    各空间维度的下界
/// @param uppers    各空间维度的上界
/// @return 0 成功，负数失败；输出文件为 ./result.h5
int dump_result_hdf5_f64_rank2(
    double* basePtr, double* data, int64_t offset,
    int64_t size0, int64_t size1,
    int64_t stride0, int64_t stride1,
    int64_t timeIndex,
    const char* const* dimNames,
    const double* lowers,
    const double* uppers);

/// 将 f64 3D memref 数据输出到 HDF5 文件
/// 参数含义同 dump_result_hdf5_f64_rank2
int dump_result_hdf5_f64_rank3(
    double* basePtr, double* data, int64_t offset,
    int64_t size0, int64_t size1, int64_t size2,
    int64_t stride0, int64_t stride1, int64_t stride2,
    int64_t timeIndex,
    const char* const* dimNames,
    const double* lowers,
    const double* uppers);

/// 将 f64 4D memref 数据输出到 HDF5 文件
/// 参数含义同 dump_result_hdf5_f64_rank2
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

#endif // EZ_COMPUTE_IO_HDF5_WRAPPER_H
