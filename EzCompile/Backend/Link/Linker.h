//===-- Linker.h -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef EZ_COMPILE_LINKER_H
#define EZ_COMPILE_LINKER_H

#include "mlir/Support/LogicalResult.h"

#include <set>
#include <filesystem>
#include <map>

#include "LinkerOptions.h"

namespace llvm {
class Module;
class Triple;
}

namespace ezcompile::link {

namespace fs = std::filesystem;

inline const fs::path manifestPath = EZ_RUNTIME_MANIFEST_ABS_PATH;
inline const fs::path baseDir = manifestPath.parent_path();

/// Manifest JSON 解析器
class ManifestParser {
public:
    /// 从指定路径加载并解析 JSON 文件
    /// @param path JSON 文件路径
    /// @param parser 输出参数，解析成功时填充
    /// @return 成功返回 success，失败返回 failure
    static mlir::LogicalResult load(const fs::path& path, ManifestParser& parser);
    
    /// 通过 key 获取字符串 value
    /// 支持嵌套 key，用 '/' 分隔，如 "archives/EzCompute.IO.HDF5"
    /// @param key 查询的 key
    /// @param value 输出参数，找到时填充
    /// @return 成功返回 success，失败返回 failure
    mlir::LogicalResult getString(const std::string& key, std::string& value) const;
    
    /// 通过 key 获取 archive 的完整路径
    /// @param key archive 的 key（如 "EzCompute.IO.HDF5"）
    /// @param archive 输出参数，完整路径（baseDir + value）
    /// @return 成功返回 success，失败返回 failure
    mlir::LogicalResult getArchivePath(const std::string& key, std::string& archive) const;
    
    /// 获取所有 archives 映射
    const std::map<std::string, std::string>& getArchives() const { return archives_; }
    
    /// 获取版本号
    int getVersion() const { return version_; }

private:
    int version_ = 0;
    std::map<std::string, std::string> archives_;
};

/// 检测到的库依赖
struct DetectedLibraries {
    std::set<std::string> libraries;
    std::set<std::string> externalFunctions;
};

/// 库检测器
class LibraryDetector {
public:
    static DetectedLibraries detect(llvm::Module &module, const std::string &targetTriple);
private:
    static bool isMathFunction(const std::string &funcName);
    static std::string getMathLibraryName(const llvm::Triple &triple);
};

/// 链接器
class Linker {
public:
    explicit Linker(const LinkerConfig &config) : config(config) {}
    
    mlir::LogicalResult run();
    
    /// 便捷方法：链接单个目标文件（自动检测需要的库）
    static mlir::LogicalResult linkModule(llvm::Module &module,
                                           const std::string &objectFile,
                                           const std::string &outputFile,
                                           const std::string &targetTriple,
                                           const std::vector<std::string> archives);

private:
    LinkerConfig config;
    
    static std::string findClang();
    std::vector<std::string> buildCommandLine() const;
};

} // namespace ezcompile::link

#endif // EZ_COMPILE_LINKER_H