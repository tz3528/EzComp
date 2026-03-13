//===-- LowerPipelines.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pass 管线实现
// 四阶段降级：Comp → Base → SCF → CF → LLVM
//
//===----------------------------------------------------------------------===//


#include "LowerPipelines.h"

namespace ezcompile {

void LowerToBase(mlir::OpPassManager &pm) {
    pm.addPass(createLowerCompDeltaPass());
    pm.addPass(createLowerCompCallPass());
    pm.addPass(createLowerCompPointsPass());
    pm.addPass(createLowerCompFieldPass());
    pm.addPass(createLowerCompApplyInitPass());
    pm.addPass(createLowerCompDirichletPass());
    pm.addPass(createLowerCompForTimePass());
    pm.addPass(createLowerCompUpdatePass());
    pm.addPass(createLowerCompSolvePass());
    pm.addPass(createLowerCompProblemPass());
    pm.addPass(createLowerCompDimPass());

    // 优化：规范化 + 常量折叠 + CSE
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());


    // 循环优化：Affine LICM（将 affine.for 内部的 loop-invariant 计算提到循环外）
    auto &fpm = pm.nest<mlir::func::FuncOp>();
    fpm.addPass(mlir::affine::createAffineLoopInvariantCodeMotionPass());
    fpm.addPass(mlir::createCanonicalizerPass());
    fpm.addPass(mlir::createCSEPass());
}

void AffineToSCF(mlir::OpPassManager &pm) {
    pm.addPass(mlir::createLowerAffinePass());
    pm.addPass(mlir::createCanonicalizerPass());
}

void SCFToCF(mlir::OpPassManager &pm) {
    pm.addPass(mlir::createSCFToControlFlowPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
}

void ToLLVM(mlir::OpPassManager &pm) {
    pm.addPass(mlir::createConvertToLLVMPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
}

} // namespace ezcompile