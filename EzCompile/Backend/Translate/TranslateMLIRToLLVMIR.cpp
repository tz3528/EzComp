//===-- TranslateMLIRToLLVMIR.cpp ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// 将 MLIR LLVM Dialect 转换为 LLVM IR
//
//===----------------------------------------------------------------------===//


#include "TranslateMLIRToLLVMIR.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

namespace ezcompile {

bool isPureLLVMModule(mlir::ModuleOp moduleOp) {
    return llvm::all_of(moduleOp.getBody()->getOperations(), [](mlir::Operation &op) {
        // 检查操作的方言是否为 LLVM 方言
        // 同时允许 builtin.func 和 builtin.module 等内置操作
        if (auto *dialect = op.getDialect()) {
            mlir::StringRef dialectName = dialect->getNamespace();
            // 允许 llvm 方言和 builtin 方言
            if (dialectName == "llvm" || dialectName == "builtin")
                return true;
        }
        // 检查操作名称是否以 llvm. 开头
        return op.getName().getStringRef().starts_with("llvm.");
    });
}

std::unique_ptr<llvm::Module> translate(mlir::ModuleOp moduleOp, llvm::LLVMContext &llvmContext) {
    // 检查 moduleOp 是否只包含 LLVM 方言
    if (!isPureLLVMModule(moduleOp)) {
        llvm::errs() << "Error: Module contains non-LLVM dialect operations. "
                     << "Please run the lowering pipeline first.\n";
        return nullptr;
    }

    // 注册 LLVM 翻译接口
    mlir::MLIRContext *mlirContext = moduleOp.getContext();
    mlir::DialectRegistry registry;
    mlir::registerBuiltinDialectTranslation(registry);
    mlir::registerLLVMDialectTranslation(registry);
    mlirContext->appendDialectRegistry(registry);

    // 将 MLIR LLVM Dialect 转换为 LLVM IR
    auto llvmModule = mlir::translateModuleToLLVMIR(moduleOp, llvmContext);
    if (!llvmModule) {
        llvm::errs() << "Error: Failed to translate MLIR to LLVM IR\n";
        return nullptr;
    }

    return llvmModule;
}

}
