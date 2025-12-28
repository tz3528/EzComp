//===-- Semantics.cpp ------------------------------------------*- C++ -*-===//
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

#include "Semantic.h"
#include "../include/Semantic.h"

namespace ezcompile {

std::unique_ptr<SemanticResult> Semantic::analyze(const ModuleAST& module) {
	collectDecls(module);
	checkOptions(module);
	checkEquations(module);
}

void Semantic::collectDecls(const ModuleAST& module) {

}

void Semantic::checkEquations(const ModuleAST& module) {

}

void Semantic::checkOptions(const ModuleAST& module) {

}

}
