//===-- CacheUtil.h ------------------------------------------- -*- C++ -*-===//
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


#ifndef EZ_RESEARCH_CACHE_UTIL_H
#define EZ_RESEARCH_CACHE_UTIL_H

#include <cstdint>
#include <optional>
#include <string>
#include <vector>
#include <algorithm>

#if defined(_WIN32)
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0601
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <sys/types.h>
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <hwloc.h>
#endif

namespace ezresearch {
/// 缓存类型
enum class CacheKind {
    /// 数据缓存
    Data,

    /// 指令缓存
    Instruction,

    /// 统一缓存
    Unified,

    /// Trace Cache，一般较少见
    Trace,

    /// 未知类型
    Unknown
};

/// 单级缓存的信息
struct CacheLevelInfo {
    /// 缓存级别，例如 1 表示 L1，2 表示 L2，3 表示 L3
    int level = 0;

    /// 缓存类型：数据缓存、指令缓存、统一缓存等
    CacheKind kind = CacheKind::Unknown;

    /// 该级缓存大小，单位：字节
    uint64_t sizeBytes = 0;

    /// 缓存行大小，单位：字节
    uint64_t lineSizeBytes = 0;

    /// 相联度：
    /// -1 表示未知或当前平台无法获取
    ///  1 表示直接映射
    /// >1 表示 N 路组相联
    ///  0 一般不作为调用方可依赖的有效值
    int associativity = -1;

    /// 是否为全相联
    bool fullyAssociative = false;

    /// 当前这一级缓存信息是否有效
    bool valid = false;
};

/// 当前机器的缓存总体信息
struct CacheInfo {
    /// 所有探测到的缓存级别信息
    std::vector<CacheLevelInfo> levels;

    /// 推荐用于数据访问优化的缓存行大小
    ///
    /// 通常优先取 L1 Data Cache 的缓存行大小；
    /// 如果没有，则退化为第一个可用缓存的缓存行大小
    uint64_t cacheLineSizeBytes = 0;

    /// 推荐用于 loop tiling 的缓存容量
    ///
    /// 通常优先取 L1 Data Cache 的大小；
    /// 如果没有，则退化为最小的 Data/Unified cache；
    /// 如果仍无法获取，则为 0
    uint64_t tilingCacheSizeBytes = 0;

    /// 当前结果是否足够可靠，可用于优化启发式
    bool valid = false;

    /// 信息来源说明，便于调试和排查
    std::string source;
};

/// 判断是否为更适合做 tiling 的缓存
inline bool isPreferredForTiling(CacheKind kind) {
    return kind == CacheKind::Data || kind == CacheKind::Unified;
}

/// 根据所有缓存级别，补全汇总字段
inline void finalizeCacheInfo(CacheInfo &info) {
    info.cacheLineSizeBytes = 0;
    info.tilingCacheSizeBytes = 0;
    info.valid = false;

    bool foundL1DataLine = false;
    bool foundAnyLine = false;

    bool foundL1DataSize = false;
    bool foundFallbackSize = false;

    for (const auto &level: info.levels) {
        if (!level.valid) {
            continue;
        }

        info.valid = true;

        /// 先确定 cache line size
        if (level.kind == CacheKind::Data && level.level == 1 &&
            level.lineSizeBytes > 0) {
            info.cacheLineSizeBytes = level.lineSizeBytes;
            foundL1DataLine = true;
        } else if (!foundL1DataLine && !foundAnyLine && level.lineSizeBytes > 0) {
            info.cacheLineSizeBytes = level.lineSizeBytes;
            foundAnyLine = true;
        }

        /// 再确定 tiling 用的 cache size
        if (level.kind == CacheKind::Data && level.level == 1 &&
            level.sizeBytes > 0) {
            info.tilingCacheSizeBytes = level.sizeBytes;
            foundL1DataSize = true;
        } else if (!foundL1DataSize && isPreferredForTiling(level.kind) && level.sizeBytes > 0) {
            if (!foundFallbackSize || level.sizeBytes < info.tilingCacheSizeBytes) {
                info.tilingCacheSizeBytes = level.sizeBytes;
                foundFallbackSize = true;
            }
        }
    }

    /// 如果没有找到 Data/Unified cache，则退化到任意 cache
    if (info.tilingCacheSizeBytes == 0) {
        for (const auto &level: info.levels) {
            if (level.valid && level.sizeBytes > 0) {
                if (info.tilingCacheSizeBytes == 0 ||
                    level.sizeBytes < info.tilingCacheSizeBytes) {
                    info.tilingCacheSizeBytes = level.sizeBytes;
                }
            }
        }
    }

    if (info.cacheLineSizeBytes == 0) {
        for (const auto &level: info.levels) {
            if (level.valid && level.lineSizeBytes > 0) {
                info.cacheLineSizeBytes = level.lineSizeBytes;
                break;
            }
        }
    }

    if (info.cacheLineSizeBytes == 0 && info.tilingCacheSizeBytes == 0) {
        info.valid = false;
    }
}

#if defined(_WIN32)

inline CacheKind toCacheKind(PROCESSOR_CACHE_TYPE type) {
    switch (type) {
        case CacheData:
            return CacheKind::Data;
        case CacheInstruction:
            return CacheKind::Instruction;
        case CacheUnified:
            return CacheKind::Unified;
        case CacheTrace:
            return CacheKind::Trace;
        default:
            return CacheKind::Unknown;
    }
}

inline std::optional<CacheInfo> detectCacheInfoImpl() {
    CacheInfo info;
    info.source = "windows:GetLogicalProcessorInformationEx";

    DWORD len = 0;
    if (!::GetLogicalProcessorInformationEx(RelationCache, nullptr, &len) &&
        ::GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
        return std::nullopt;
    }

    if (len == 0) {
        return std::nullopt;
    }

    std::vector<unsigned char> buffer(len);
    auto *ptrInfo = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data());

    if (!::GetLogicalProcessorInformationEx(RelationCache, ptrInfo, &len)) {
        return std::nullopt;
    }

    unsigned char *ptr = buffer.data();
    unsigned char *end = buffer.data() + len;

    while (ptr < end) {
        auto *entry = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(ptr);

        if (entry->Relationship == RelationCache) {
            const CACHE_RELATIONSHIP &c = entry->Cache;

            CacheLevelInfo levelInfo;
            levelInfo.level = static_cast<int>(c.Level);
            levelInfo.kind = toCacheKind(c.Type);
            levelInfo.sizeBytes = static_cast<uint64_t>(c.CacheSize);
            levelInfo.lineSizeBytes = static_cast<uint64_t>(c.LineSize);
            levelInfo.fullyAssociative = (c.Associativity == CACHE_FULLY_ASSOCIATIVE);

            if (levelInfo.fullyAssociative) {
                levelInfo.associativity = -1;
            } else {
                levelInfo.associativity = static_cast<int>(c.Associativity);
            }

            levelInfo.valid = (levelInfo.level > 0) && (levelInfo.sizeBytes > 0 || levelInfo.lineSizeBytes > 0);

            if (levelInfo.valid) {
                info.levels.push_back(levelInfo);
            }
        }

        ptr += entry->Size;
    }

    finalizeCacheInfo(info);
    if (!info.valid) {
        return std::nullopt;
    }
    return info;
}

#elif defined(__APPLE__)

template<typename T>
inline bool getSysctlValue(const char *name, T &value) {
    size_t size = sizeof(T);
    return ::sysctlbyname(name, &value, &size, nullptr, 0) == 0;
}

inline void tryAddAppleCacheLevel(CacheInfo &info,
                                  int level,
                                  CacheKind kind,
                                  const char *sizeName,
                                  uint64_t lineSize) {
    uint64_t sizeValue = 0;
    if (!getSysctlValue(sizeName, sizeValue) || sizeValue == 0) {
        return;
    }

    CacheLevelInfo levelInfo;
    levelInfo.level = level;
    levelInfo.kind = kind;
    levelInfo.sizeBytes = sizeValue;
    levelInfo.lineSizeBytes = lineSize;
    levelInfo.associativity = -1;
    levelInfo.fullyAssociative = false;
    levelInfo.valid = true;

    info.levels.push_back(levelInfo);
}

inline std::optional<CacheInfo> detectCacheInfoImpl() {
    CacheInfo info;
    info.source = "macos:sysctlbyname";

    uint64_t lineSize = 0;
    getSysctlValue("hw.cachelinesize", lineSize);

    /// L1I
    tryAddAppleCacheLevel(info, 1, CacheKind::Instruction, "hw.l1icachesize", lineSize);

    /// L1D
    tryAddAppleCacheLevel(info, 1, CacheKind::Data, "hw.l1dcachesize", lineSize);

    /// L2
    tryAddAppleCacheLevel(info, 2, CacheKind::Unified, "hw.l2cachesize", lineSize);

    /// L3
    tryAddAppleCacheLevel(info, 3, CacheKind::Unified, "hw.l3cachesize", lineSize);

    finalizeCacheInfo(info);
    if (!info.valid) {
        return std::nullopt;
    }
    return info;
}

#elif defined(__linux__)

inline CacheKind toCacheKind(hwloc_obj_cache_type_t type) {
    switch (type) {
        case HWLOC_OBJ_CACHE_DATA:
            return CacheKind::Data;
        case HWLOC_OBJ_CACHE_INSTRUCTION:
            return CacheKind::Instruction;
        case HWLOC_OBJ_CACHE_UNIFIED:
            return CacheKind::Unified;
        default:
            return CacheKind::Unknown;
    }
}

inline bool isCacheObjectType(hwloc_obj_type_t type) {
    return type == HWLOC_OBJ_L1CACHE ||
           type == HWLOC_OBJ_L2CACHE ||
           type == HWLOC_OBJ_L3CACHE ||
           type == HWLOC_OBJ_L4CACHE ||
           type == HWLOC_OBJ_L5CACHE;
}

inline std::optional<CacheInfo> detectCacheInfoImpl() {
    CacheInfo info;
    info.source = "linux:hwloc";

    hwloc_topology_t topo;
    if (hwloc_topology_init(&topo) != 0) {
        return std::nullopt;
    }

    if (hwloc_topology_load(topo) != 0) {
        hwloc_topology_destroy(topo);
        return std::nullopt;
    }

    int depthCount = hwloc_topology_get_depth(topo);
    for (int depth = 0; depth < depthCount; ++depth) {
        int count = hwloc_get_nbobjs_by_depth(topo, depth);
        for (int i = 0; i < count; ++i) {
            hwloc_obj_t obj = hwloc_get_obj_by_depth(topo, depth, i);
            if (!obj || !obj->attr) {
                continue;
            }
            if (!isCacheObjectType(obj->type)) {
                continue;
            }

            const auto &c = obj->attr->cache;

            CacheLevelInfo levelInfo;
            levelInfo.level = static_cast<int>(c.depth);
            levelInfo.kind = toCacheKind(c.type);
            levelInfo.sizeBytes = static_cast<uint64_t>(c.size);
            levelInfo.lineSizeBytes = static_cast<uint64_t>(c.linesize);
            levelInfo.associativity = (c.associativity > 0) ? static_cast<int>(c.associativity) : -1;
            levelInfo.fullyAssociative = false;
            levelInfo.valid = (levelInfo.level > 0) && (levelInfo.sizeBytes > 0 || levelInfo.lineSizeBytes > 0);

            if (levelInfo.valid) {
                info.levels.push_back(levelInfo);
            }
        }
    }

    hwloc_topology_destroy(topo);

    finalizeCacheInfo(info);
    if (!info.valid) {
        return std::nullopt;
    }
    return info;
}

#else

inline std::optional<CacheInfo> detectCacheInfoImpl() {
    return std::nullopt;
}

#endif

/// 探测当前机器的缓存信息
///
/// 该函数已在当前头文件内直接实现，支持跨平台：
/// - Windows
/// - Linux（依赖 hwloc）
/// - macOS
///
/// 成功时返回 CacheInfo；
/// 失败时返回 std::nullopt
inline std::optional<CacheInfo> detectCacheInfo() {
    return detectCacheInfoImpl();
}

/// 仅获取优化 pass 常用的两个参数：CL 和 C
///
/// 参数：
///   CL       输出缓存行大小
///   C        输出推荐用于 tiling 的缓存大小
///   fullInfo 如果非空，则同时返回完整的 CacheInfo
///
/// 返回值：
///   true  表示获取成功
///   false 表示获取失败
inline bool getTilingCacheParams(uint64_t &CL, uint64_t &C, CacheInfo *fullInfo = nullptr) {
    auto info = detectCacheInfo();
    if (!info || !info->valid) {
        return false;
    }

    CL = info->cacheLineSizeBytes;
    C = info->tilingCacheSizeBytes;

    if (fullInfo) {
        *fullInfo = *info;
    }

    return true;
}

}

#endif //EZ_RESEARCH_CACHE_UTIL_H
