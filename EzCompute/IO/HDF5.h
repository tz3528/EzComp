//===-- HDF5.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//
//===----------------------------------------------------------------------===//


#ifndef EZ_COMPUTE_IO_HDF5_H
#define EZ_COMPUTE_IO_HDF5_H

#include <cstdint>

namespace ezcompute {

// 这是 MLIR lowering 到 LLVM 时常用的“strided memref descriptor”形态：
// - strides/sizes 单位是“元素个数”（不是字节）
// - data 是对齐后的可用指针；offset 是逻辑偏移
template <typename T, int Rank>
struct StridedMemRefType {
    T* basePtr;
    T* data;
    int64_t offset;
    int64_t sizes[Rank];
    int64_t strides[Rank];
};

#ifdef __cplusplus
extern "C" {
#endif

// 说明：
// - Rank 是 memref 的维数：第 0 维是 time-layer（例如 2），后面 (Rank-1) 维是空间维。
// - timeIndex 是你传入的时间下标 n（或 layer 也行），内部用 layer = timeIndex % sizes[0] 选择要输出的那层。
// - dimNames/lowers/uppers 只描述“空间维”(Rank-1 个)，顺序保证与你内存维度使用顺序一致。
// - points 不需要额外传：直接用 memref 的 sizes[1..]。
// - 输出：当前工作目录下的 result.h5，覆盖旧文件；主数据集路径是 "/result"。
//
// 返回：0 成功；负数失败。
int dump_result_hdf5_f64_rank2(const StridedMemRefType<double, 2>* memref,
                               int64_t timeIndex,
                               const char* const* dimNames,
                               const double* lowers,
                               const double* uppers);

int dump_result_hdf5_f64_rank3(const StridedMemRefType<double, 3>* memref,
                               int64_t timeIndex,
                               const char* const* dimNames,
                               const double* lowers,
                               const double* uppers);

int dump_result_hdf5_f64_rank4(const StridedMemRefType<double, 4>* memref,
                               int64_t timeIndex,
                               const char* const* dimNames,
                               const double* lowers,
                               const double* uppers);

#ifdef __cplusplus
} // extern "C"s
#endif

}

#endif // EZ_COMPUTE_IO_HDF5_H
