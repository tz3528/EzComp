//===-- BakendDriver.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// 后端编译驱动
//
//===----------------------------------------------------------------------===//


#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/TargetParser/Host.h"

#include "BackendDriver.h"
#include "Target/CodeEmitter.h"
#include "Link/Linker.h"

namespace ezcompile {

void Backend::dumpLLVMIR(llvm::Module &module) {
    llvm::errs() << "=== LLVM IR ===\n";
    module.print(llvm::outs(), nullptr);
    llvm::errs() << "\n=== End LLVM IR ===\n";
}

std::vector<std::string> Backend::getRequiredArchives() {
    std::vector<std::string> archives;
    archives.push_back("IO.HDF5");
    return archives;
}

mlir::LogicalResult Backend::codeGen(llvm::Module &module,
                                      llvm::TargetMachine &targetMachine,
                                      const std::string &outputPath) {
    target::CodeEmitter emitter(targetMachine);
    return emitter.emit(module, outputPath, target::OutputFileType::ObjectFile);
}

mlir::LogicalResult Backend::fullCompile(llvm::Module &module) {
    // 获取默认目标三元组
    std::string tripleStr = llvm::sys::getDefaultTargetTriple();

    // 初始化目标
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmParsers();
    llvm::InitializeAllAsmPrinters();

    // 查找目标
    std::string error;
    const llvm::Target *target = llvm::TargetRegistry::lookupTarget(tripleStr, error);
    if (!target) {
        llvm::errs() << "Error looking up target: " << error << "\n";
        return mlir::failure();
    }

    // 创建 TargetMachine
    llvm::TargetOptions opt;
    std::optional<llvm::Reloc::Model> rm = llvm::Reloc::PIC_;
    std::string cpu = config.targetCPUVal.empty() ? "generic" : config.targetCPUVal;

    llvm::Triple triple(tripleStr);
    std::unique_ptr<llvm::TargetMachine> targetMachine(
        target->createTargetMachine(triple, cpu, config.targetFeaturesVal, opt, rm));

    if (!targetMachine) {
        llvm::errs() << "Error creating target machine\n";
        return mlir::failure();
    }

    // 确定输出文件路径
    std::string objFile = config.outputFileVal.empty() ? "a.o" : config.outputFileVal + ".o";
    std::string exeFile = config.outputFileVal.empty() ? "a.out" : config.outputFileVal;

    // 代码生成
    if (mlir::failed(codeGen(module, *targetMachine, objFile))) {
        return mlir::failure();
    }

    // 链接
    if (mlir::failed(link::Linker::linkModule(module, objFile, exeFile, getRequiredArchives()))) {
        return mlir::failure();
    }

    llvm::errs() << "Output: " << exeFile << "\n";
    return mlir::success();
}

mlir::LogicalResult Backend::run(mlir::ModuleOp &module) {
    llvm::LLVMContext llvmContext;
    auto llvmModule = translate(module, llvmContext);
    if (!llvmModule) return mlir::failure();

    switch (config.mode) {
    case backend::CompileMode::DumpLLVMIR:
        dumpLLVMIR(*llvmModule);
        return mlir::success();
    case backend::CompileMode::FullCompile:
        return fullCompile(*llvmModule);
    }
    return mlir::failure();
}

}