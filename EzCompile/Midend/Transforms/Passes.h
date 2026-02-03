//===-- Passes.h -----------------------------------------------*- C++ -*-===//
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


#ifndef EZ_COMPILE_PASSES_H
#define EZ_COMPILE_PASSES_H

#include "mlir/Pass/Pass.h"

namespace ezcompile {

void registerLowerCompPointsToArithPass();

void registerPasses() {
	registerLowerCompPointsToArithPass();
}

}

#endif //EZ_COMPILE_PASSES_H
