//===-- LowerPasses.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pass 注册接口
// 声明所有降级 Pass 的注册和创建函数
//
//===----------------------------------------------------------------------===//


#ifndef EZ_COMPILE_LOWER_PASSES_H
#define EZ_COMPILE_LOWER_PASSES_H

#include "mlir/Pass/Pass.h"

namespace ezcompile {

//===----------------------------------------------------------------------===//
// Comp 方言降级 Pass 注册
//===----------------------------------------------------------------------===//

void registerLowerCompDimPass();
void registerLowerCompFieldPass();
void registerLowerCompPointsPass();
void registerLowerCompApplyInitPass();
void registerLowerCompForTimePass();
void registerLowerCompDirichletPass();
void registerLowerCompUpdatePass();
void registerLowerCompDimPass();
void registerLowerCompProblemPass();
void registerLowerCompCallPass();
void registerLowerCompDeltaPass();

/// 注册所有 Pass
inline void registerPasses() {
	registerLowerCompDimPass();
	registerLowerCompFieldPass();
	registerLowerCompPointsPass();
	registerLowerCompApplyInitPass();
	registerLowerCompForTimePass();
	registerLowerCompDirichletPass();
	registerLowerCompUpdatePass();
	registerLowerCompDimPass();
	registerLowerCompProblemPass();
	registerLowerCompCallPass();
	registerLowerCompDeltaPass();
}

}

#endif //EZ_COMPILE_LOWER_PASSES_H
