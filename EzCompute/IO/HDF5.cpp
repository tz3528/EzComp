//===-- HDF5.cpp -----------------------------------------------*- C++ -*-===//
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


#include <hdf5.h>
#include <cctype>
#include <cstring>
#include <string>
#include <vector>

#include "HDF5.h"

namespace ezcompute {

// 把维度名做一个安全化：只保留 [A-Za-z0-9_]，其它字符替换成 '_'。
// 避免 HDF5 路径名/属性名出现奇怪字符。
static std::string sanitizeName(const char* s, int fallbackIdx) {
    if (!s || !*s) return "dim" + std::to_string(fallbackIdx);
    std::string out;
    for (const unsigned char* p = (const unsigned char*)s; *p; ++p) {
        if (std::isalnum(*p) || *p == '_') out.push_back((char)*p);
        else out.push_back('_');
    }
    if (out.empty()) out = "dim" + std::to_string(fallbackIdx);
    return out;
}

// 写一个 int64 的 attribute（标量）。
static bool write_attr_i64(hid_t obj, const char* name, int64_t v) {
    hid_t space = H5Screate(H5S_SCALAR);
    if (space < 0) return false;
    hid_t attr = H5Acreate2(obj, name, H5T_NATIVE_INT64, space, H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0) { H5Sclose(space); return false; }
    herr_t st = H5Awrite(attr, H5T_NATIVE_INT64, &v);
    H5Aclose(attr);
    H5Sclose(space);
    return st >= 0;
}

// 写一个 double 的 attribute（标量）。
static bool write_attr_f64(hid_t obj, const char* name, double v) {
    hid_t space = H5Screate(H5S_SCALAR);
    if (space < 0) return false;
    hid_t attr = H5Acreate2(obj, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0) { H5Sclose(space); return false; }
    herr_t st = H5Awrite(attr, H5T_NATIVE_DOUBLE, &v);
    H5Aclose(attr);
    H5Sclose(space);
    return st >= 0;
}

// 写一个字符串数组 attribute（dim_names）。
static bool write_attr_strs(hid_t obj, const char* name, const std::vector<std::string>& vals) {
    if (vals.empty()) return true;

    hid_t vlenStr = H5Tcopy(H5T_C_S1);
    if (vlenStr < 0) return false;
    H5Tset_size(vlenStr, H5T_VARIABLE);

    hsize_t dims[1] = { static_cast<hsize_t>(vals.size()) };
    hid_t space = H5Screate_simple(1, dims, nullptr);
    if (space < 0) { H5Tclose(vlenStr); return false; }

    hid_t attr = H5Acreate2(obj, name, vlenStr, space, H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0) { H5Sclose(space); H5Tclose(vlenStr); return false; }

    std::vector<const char*> cstrs;
    cstrs.reserve(vals.size());
    for (auto& s : vals) cstrs.push_back(s.c_str());

    herr_t st = H5Awrite(attr, vlenStr, cstrs.data());

    H5Aclose(attr);
    H5Sclose(space);
    H5Tclose(vlenStr);
    return st >= 0;
}

// 写一个 1D 坐标数据集（不使用 group）：例如 "/coord_x"。
static bool write_coords_1d(hid_t file,
                            const std::string& dsName,
                            int64_t points,
                            double lower,
                            double upper) {
    if (points <= 0) return false;

    std::vector<double> coords(static_cast<size_t>(points));
    if (points == 1) {
        coords[0] = lower;
    } else {
        // 包含端点的 linspace
        double step = (upper - lower) / static_cast<double>(points - 1);
        for (int64_t i = 0; i < points; ++i) {
            coords[static_cast<size_t>(i)] = lower + step * static_cast<double>(i);
        }
    }

    hsize_t dims[1] = { static_cast<hsize_t>(points) };
    hid_t space = H5Screate_simple(1, dims, nullptr);
    if (space < 0) return false;

    // 若已存在同名数据集，先删掉（因为文件是 TRUNC 理论上不会存在，但写稳一点）。
    H5Ldelete(file, dsName.c_str(), H5P_DEFAULT);

    hid_t dset = H5Dcreate2(file, dsName.c_str(), H5T_NATIVE_DOUBLE, space,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dset < 0) { H5Sclose(space); return false; }

    herr_t st = H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, coords.data());

    H5Dclose(dset);
    H5Sclose(space);
    return st >= 0;
}

// 模板实现：Rank = memref rank（time-layer + space dims）
template <int Rank>
static int dump_impl(const StridedMemRefType<double, Rank>* m,
                     int64_t timeIndex,
                     const char* const* dimNames,
                     const double* lowers,
                     const double* uppers) {
    static_assert(Rank >= 2, "memref 至少需要 [timeLayer, space...] 两维");

    if (!m || !m->data) return -1;

    const int64_t timeLayers = m->sizes[0];
    if (timeLayers <= 0) return -2;

    // 你传入的是时间下标 n：这里统一映射到 layer（对 2-layer 就是 mod 2）
    int64_t layer = timeIndex % timeLayers;
    if (layer < 0) layer += timeLayers;

    constexpr int SRank = Rank - 1; // 空间维数
    if (!dimNames || !lowers || !uppers) return -3;

    // 计算空间总元素数、以及 HDF5 维度数组
    int64_t spatialCount = 1;
    hsize_t dims[SRank];
    for (int i = 0; i < SRank; ++i) {
        int64_t sz = m->sizes[i + 1];
        if (sz <= 0) return -4;
        spatialCount *= sz;
        dims[i] = static_cast<hsize_t>(sz);
    }

    // 把 slice 收集成连续内存，方便一次性写入 HDF5。
    // idx = offset + layer*stride0 + Σ(iv[k]*stride[k+1])
    std::vector<double> out(static_cast<size_t>(spatialCount));
    std::vector<int64_t> iv(SRank, 0);

    for (int64_t linear = 0; linear < spatialCount; ++linear) {
        int64_t idx = m->offset + layer * m->strides[0];
        for (int k = 0; k < SRank; ++k) {
            idx += iv[k] * m->strides[k + 1];
        }
        out[static_cast<size_t>(linear)] = m->data[idx];

        // 以“最后一维最快”的顺序递增 iv（与常见 row-major 一致）
        for (int k = SRank - 1; k >= 0; --k) {
            iv[k]++;
            if (iv[k] < m->sizes[k + 1]) break;
            iv[k] = 0;
        }
    }

    // 创建/覆盖输出文件：当前工作目录下 result.h5
    hid_t file = H5Fcreate("result.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file < 0) return -5;

    // 创建结果数据集：/result
    hid_t space = H5Screate_simple(SRank, dims, nullptr);
    if (space < 0) { H5Fclose(file); return -6; }

    hid_t dset = H5Dcreate2(file, "/result", H5T_NATIVE_DOUBLE, space,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dset < 0) { H5Sclose(space); H5Fclose(file); return -7; }

    if (H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, out.data()) < 0) {
        H5Dclose(dset); H5Sclose(space); H5Fclose(file); return -8;
    }

    // =========================
    // 写入“维度信息”（不使用 group）
    // =========================

    // 记录这次输出对应的 timeIndex 和映射后的 layer
    (void)write_attr_i64(dset, "time_index", timeIndex);
    (void)write_attr_i64(dset, "layer_index", layer);
    (void)write_attr_i64(dset, "space_rank", SRank);

    // 维度名数组 attribute：dim_names = ["x","y",...]
    std::vector<std::string> safeDimNames;
    safeDimNames.reserve(SRank);
    for (int i = 0; i < SRank; ++i) {
        safeDimNames.push_back(sanitizeName(dimNames[i], i));
    }
    (void)write_attr_strs(dset, "dim_names", safeDimNames);

    // 每个维度的 lower/upper/points 写成 attribute：
    // 例如：x.lower / x.upper / x.points
    for (int i = 0; i < SRank; ++i) {
        const std::string& nm = safeDimNames[i];
        const int64_t points = m->sizes[i + 1];
        const double lo = lowers[i];
        const double up = uppers[i];

        (void)write_attr_i64(dset, (nm + ".points").c_str(), points);
        (void)write_attr_f64(dset, (nm + ".lower").c_str(), lo);
        (void)write_attr_f64(dset, (nm + ".upper").c_str(), up);

        // 同时写一个坐标数据集（不使用 group）：/coord_<name>
        // 坐标规则：包含端点的 linspace(lower, upper, points)
        (void)write_coords_1d(file, "/coord_" + nm, points, lo, up);
    }

    H5Dclose(dset);
    H5Sclose(space);
    H5Fclose(file);
    return 0;
}

// =========================
// 对外导出的 C ABI 包装函数
// =========================
extern "C" int dump_result_hdf5_f64_rank2(const StridedMemRefType<double, 2>* memref,
                                          int64_t timeIndex,
                                          const char* const* dimNames,
                                          const double* lowers,
                                          const double* uppers) {
    return dump_impl<2>(memref, timeIndex, dimNames, lowers, uppers);
}

extern "C" int dump_result_hdf5_f64_rank3(const StridedMemRefType<double, 3>* memref,
                                          int64_t timeIndex,
                                          const char* const* dimNames,
                                          const double* lowers,
                                          const double* uppers) {
    return dump_impl<3>(memref, timeIndex, dimNames, lowers, uppers);
}

extern "C" int dump_result_hdf5_f64_rank4(const StridedMemRefType<double, 4>* memref,
                                          int64_t timeIndex,
                                          const char* const* dimNames,
                                          const double* lowers,
                                          const double* uppers) {
    return dump_impl<4>(memref, timeIndex, dimNames, lowers, uppers);
}

}
