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
#include "llvm/Support/raw_ostream.h"

#include "BakendDriver.h"

namespace ezcompile {

mlir::LogicalResult Bakend::run(mlir::ModuleOp &module) {
    // 1. 创建 LLVM Context
    llvm::LLVMContext llvmContext;

    // 2. 将 MLIR 转换为 LLVM IR
    auto llvmModule = translate(module, llvmContext);
    if (!llvmModule) {
        // 错误已在 translate 内部报告
        return mlir::failure();
    }

    // Dump LLVM IR
    llvm::errs() << "=== LLVM IR ===\n";
    llvmModule->print(llvm::errs(), nullptr);
    llvm::errs() << "=== End LLVM IR ===\n";

    // TODO: 3. Target - 代码生成
    // TODO: 4. Link - 链接

    return mlir::success();
}

}
