//===-- CodeEmitter.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// 代码发射实现：LLVM IR -> 目标代码
//
//===----------------------------------------------------------------------===//


#include "CodeEmitter.h"

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

namespace ezcompile::target {

CodeEmitter::CodeEmitter(llvm::TargetMachine &targetMachine)
    : targetMachine(targetMachine),
      targetInfo(std::make_unique<TargetInfo>(targetMachine)) {}

mlir::LogicalResult CodeEmitter::emit(llvm::Module &module,
                                       const std::string &outputPath,
                                       OutputFileType fileType) {
    // 1. 打开输出文件
    llvm::sys::fs::OpenFlags flags = (fileType == OutputFileType::AssemblyFile)
        ? llvm::sys::fs::OF_Text : llvm::sys::fs::OF_None;
    
    std::error_code ec;
    llvm::raw_fd_ostream os(outputPath, ec, flags);
    if (ec) {
        llvm::errs() << "Error opening output file '" << outputPath
                     << "': " << ec.message() << "\n";
        return mlir::failure();
    }

    // 2. 配置模块目标信息
    module.setDataLayout(targetMachine.createDataLayout());
    module.setTargetTriple(targetMachine.getTargetTriple());

    // 3. 创建 PassManager 并发射代码
    llvm::legacy::PassManager passManager;
    llvm::CodeGenFileType cgFileType = (fileType == OutputFileType::ObjectFile)
        ? llvm::CodeGenFileType::ObjectFile
        : llvm::CodeGenFileType::AssemblyFile;

    if (targetMachine.addPassesToEmitFile(passManager, os, nullptr, cgFileType)) {
        llvm::errs() << "TargetMachine cannot emit a file of this type\n";
        return mlir::failure();
    }

    passManager.run(module);
    return mlir::success();
}

} // namespace ezcompile::target
