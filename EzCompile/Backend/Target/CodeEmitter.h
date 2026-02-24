//===-- CodeEmitter.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// 代码发射接口：LLVM IR -> 目标代码
//
//===----------------------------------------------------------------------===//


#ifndef EZ_COMPILE_CODE_EMITTER_H
#define EZ_COMPILE_CODE_EMITTER_H

#include "TargetInfo.h"

#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Support/LogicalResult.h"

#include <memory>
#include <string>

namespace ezcompile::target {

/// 输出文件类型
enum class OutputFileType {
    ObjectFile,     // .o 目标文件
    AssemblyFile,   // .s 汇编文件
};

/// 代码发射器
/// 将 LLVM IR 转换为目标代码（汇编或目标文件）
class CodeEmitter {
public:
    explicit CodeEmitter(llvm::TargetMachine &targetMachine);

    /// 发射代码到文件
    /// @param module LLVM IR 模块
    /// @param outputPath 输出文件路径
    /// @param fileType 文件类型（目标文件/汇编文件）
    /// @return 成功或失败
    mlir::LogicalResult emit(llvm::Module &module,
                              const std::string &outputPath,
                              OutputFileType fileType);

    /// 获取目标信息
    const TargetInfo &getTargetInfo() const { return *targetInfo; }

private:
    llvm::TargetMachine &targetMachine;
    std::unique_ptr<TargetInfo> targetInfo;
};

} // namespace ezcompile::target

#endif // EZ_COMPILE_CODE_EMITTER_H
