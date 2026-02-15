//===-- CompDialect.cpp ----------------------------------------*- C++ -*-===//
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

#include "CompDialect.h"
#include "CompOps.h"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"

#include "CompDialect.cpp.inc"

namespace ezcompile::comp {
using ::mlir::Type;
using ::mlir::Attribute;
using ::mlir::Builder;
}


#define GET_TYPEDEF_CLASSES
#include "CompTypes.cpp.inc"
#undef GET_TYPEDEF_CLASSES

#define GET_ATTRDEF_CLASSES
#include "CompAttributes.cpp.inc"
#undef GET_ATTRDEF_CLASSES

void ezcompile::comp::CompDialect::initialize() {
	addOperations<
#define GET_OP_LIST
#include "CompOps.cpp.inc"
#undef GET_OP_LIST
	>();

	addTypes<
#define GET_TYPEDEF_LIST
#include "CompTypes.cpp.inc"
#undef GET_TYPEDEF_LIST
	>();

	addAttributes<
#define GET_ATTRDEF_LIST
#include "CompAttributes.cpp.inc"
#undef GET_ATTRDEF_LIST
	>();
}
