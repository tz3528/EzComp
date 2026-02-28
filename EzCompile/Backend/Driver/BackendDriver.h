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

    mlir::LogicalResult run(mlir::ModuleOp &module);

private:
    backend::BackendConfig config;

    mlir::LogicalResult fullCompile(llvm::Module &module);
    mlir::LogicalResult codeGen(llvm::Module &module,
                                 llvm::TargetMachine &targetMachine,
                                 const std::string &outputPath);
    void dumpLLVMIR(llvm::Module &module);
};

}

#endif //EZ_COMPILE_BAKEND_DRIVER_H