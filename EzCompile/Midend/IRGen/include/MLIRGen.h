//===-- MLIRGen.h ----------------------------------------------*- C++ -*-===//
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


#ifndef EZ_COMPILE_MLIR_GEN_H
#define EZ_COMPILE_MLIR_GEN_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Value.h"

#include <memory>

#include "Semantic.h"
#include "Dialects/Comp/include/Comp.h"

namespace ezcompile {

class MLIRGen {
	explicit MLIRGen(std::unique_ptr<ParsedModule> pm);

	mlir::ModuleOp mlirGen();
private:
	std::unique_ptr<ParsedModule> pm;
	mlir::OpBuilder builder;
};

}

#endif //EZ_COMPILE_MLIR_GEN_H
