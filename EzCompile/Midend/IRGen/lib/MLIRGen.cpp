//===-- MLIRGen.cpp --------------------------------------------*- C++ -*-===//
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

#include "IRGen/include/MLIRGen.h"

namespace ezcompile {

MLIRGen::MLIRGen(std::unique_ptr<ParsedModule> pm)
	: pm(std::move(pm)), builder(&pm->context) {

}


}