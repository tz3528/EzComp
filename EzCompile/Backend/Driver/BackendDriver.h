//===-- BakendDriver.h -----------------------------------------*- C++ -*-===//
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


#ifndef EZ_COMPILE_BAKEND_DRIVER_H
#define EZ_COMPILE_BAKEND_DRIVER_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include "Translate/TranslateMLIRToLLVMIR.h"
#include "BackendOptions.h"

#include <memory>

namespace llvm {
class Module;
class TargetMachine;
}

namespace ezcompile {

class Backend {
public:
    explicit Backend(const backend::BackendConfig &config = backend::BackendConfig::forDumpLLVMIR())
        : config(config) {}

    /// 执行后端编译流程
    mlir::LogicalResult run(mlir::ModuleOp &module);

private:
    backend::BackendConfig config;

    /// 完整编译流程（代码生成 + 链接）
    mlir::LogicalResult fullCompile(llvm::Module &module);

    /// 代码生成 (LLVM IR -> 目标代码)
    mlir::LogicalResult codeGen(llvm::Module &module, llvm::TargetMachine &targetMachine);

    /// 链接生成可执行文件
    mlir::LogicalResult link(const std::string &objectFile, const std::string &outputFile);

    /// 输出 LLVM IR 到标准输出
    void dumpLLVMIR(llvm::Module &module);
};

}

#endif //EZ_COMPILE_BAKEND_DRIVER_H
