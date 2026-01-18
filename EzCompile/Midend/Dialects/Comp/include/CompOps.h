//===-- CompOps.h ----------------------------------------------*- C++ -*-===//
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


#ifndef EZ_COMPILE_COMP_OPS_H
#define EZ_COMPILE_COMP_OPS_H

#include "mlir/IR/SymbolTable.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "CompDialect.h"

#define GET_TYPEDEF_CLASSES
#include "CompTypes.h.inc"
#undef GET_TYPEDEF_CLASSES

#define GET_ATTRDEF_CLASSES
#include "CompAttributes.h.inc"
#undef GET_ATTRDEF_CLASSES

#define GET_OP_CLASSES
#include "CompOps.h.inc"
#undef GET_OP_CLASSES

#endif //EZ_COMPILE_COMP_OPS_H
