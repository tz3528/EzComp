//===-- TranslateMLIRToLLVMIR.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// 
//
//===----------------------------------------------------------------------===//


#ifndef EZ_COMPILE_TRANSLATE_MLIR_TO_LLVMIR_H
#define EZ_COMPILE_TRANSLATE_MLIR_TO_LLVMIR_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

namespace ezcompile {

std::unique_ptr<llvm::Module> translate(mlir::ModuleOp moduleOp, llvm::LLVMContext &llvmContext);

}

#endif //EZ_COMPILE_TRANSLATE_MLIR_TO_LLVMIR_H
