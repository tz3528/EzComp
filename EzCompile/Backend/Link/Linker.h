//===-- Linker.h -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef EZ_COMPILE_LINKER_H
#define EZ_COMPILE_LINKER_H

#include "LinkerOptions.h"

#include "mlir/Support/LogicalResult.h"

#include <set>

namespace llvm {
class Module;
class Triple;
}

namespace ezcompile::link {

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
                                           const std::string &targetTriple);

private:
    LinkerConfig config;
    
    static std::string findClang();
    std::vector<std::string> buildCommandLine() const;
};

} // namespace ezcompile::link

#endif // EZ_COMPILE_LINKER_H