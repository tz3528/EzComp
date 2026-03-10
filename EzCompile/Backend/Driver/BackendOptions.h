//===-- BackendOptions.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// 后端编译选项定义
//
//===----------------------------------------------------------------------===//


#ifndef EZ_COMPILE_BACKEND_OPTIONS_H
#define EZ_COMPILE_BACKEND_OPTIONS_H

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Path.h"
#include <string>

namespace ezcompile::backend {

namespace cl = llvm::cl;

//===----------------------------------------------------------------------===//
// 编译模式
//===----------------------------------------------------------------------===//

/// 后端编译模式
enum class CompileMode {
    DumpLLVMIR,     // 仅输出 LLVM IR
    FullCompile     // 完整编译到可执行文件
};

//===----------------------------------------------------------------------===//
// 代码生成选项
//===----------------------------------------------------------------------===//

inline cl::opt<std::string> targetCPU(
    "mcpu",
    cl::desc("Target CPU"),
    cl::value_desc("cpu"));

inline cl::opt<std::string> targetFeatures(
    "mattr",
    cl::desc("Target CPU features (e.g. +avx2,+fma)"),
    cl::value_desc("features"));

//===----------------------------------------------------------------------===//
// 输出选项
//===----------------------------------------------------------------------===//

inline cl::opt<std::string> outputFile(
    "o",
    cl::desc("Output file name"),
    cl::value_desc("filename"));

//===----------------------------------------------------------------------===//
// 后端配置结构（运行时配置，用于传递给 Backend）
//===----------------------------------------------------------------------===//

struct BackendConfig {
    CompileMode mode = CompileMode::DumpLLVMIR;
    std::string targetCPUVal;
    std::string targetFeaturesVal;
    std::string outputFileVal;

    /// 从命令行选项创建配置（用于 FullCompile 模式）
    /// @param inputFile 输入文件名，用于生成默认输出文件名
    static BackendConfig fromCommandLine(llvm::StringRef inputFile = "");

    /// 创建 DumpLLVMIR 模式的默认配置
    static BackendConfig forDumpLLVMIR() {
        BackendConfig config;
        config.mode = CompileMode::DumpLLVMIR;
        return config;
    }
};

inline BackendConfig BackendConfig::fromCommandLine(llvm::StringRef inputFile) {
    BackendConfig config;
    config.mode = CompileMode::FullCompile;
    config.targetCPUVal = targetCPU.getValue();
    config.targetFeaturesVal = targetFeatures.getValue();
    
    // 如果用户未指定输出文件名，则基于输入文件名生成默认值
    if (outputFile.empty() && !inputFile.empty()) {
        llvm::StringRef baseName = inputFile;
        // 去掉 .comp 后缀
        if (baseName.ends_with(".comp")) {
            baseName = baseName.drop_back(5);
        }
        // 去掉路径，只保留文件名
        baseName = llvm::sys::path::filename(baseName);
        config.outputFileVal = baseName.str();
    } else {
        config.outputFileVal = outputFile.getValue();
    }
    return config;
}

} // namespace ezcompile::backend

#endif // EZ_COMPILE_BACKEND_OPTIONS_H