//===-- OptPipelines.cpp -------------------------------------- -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
///
//===----------------------------------------------------------------------===//


#include "Transforms/OptPipelines.h"

#include "OptPasses.h"

namespace ezresearch {

void HoistBoundary(mlir::OpPassManager &pm) {
    pm.addPass(createOptBoundaryHoistPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
}

void AffineVectorize(mlir::OpPassManager &pm){
    auto &fpm = pm.nest<mlir::func::FuncOp>();
    mlir::affine::AffineVectorizeOptions opts;
    opts.vectorSizes = {4};              // 先试最内层 x 方向 4-wide
    opts.vectorizeReductions = false;    // 你这个 stencil 先不用管 reduction

    fpm.addPass(mlir::affine::createAffineVectorize(opts));

    fpm.addPass(mlir::createCanonicalizerPass());
    fpm.addPass(mlir::createCSEPass());
    fpm.addPass(mlir::affine::createAffineLoopInvariantCodeMotionPass());
    fpm.addPass(mlir::createCanonicalizerPass());
    fpm.addPass(mlir::createCSEPass());
}

void LoopPeeling(mlir::OpPassManager &pm) {
    pm.addPass(createOptLoopPeelingPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
}

void LoopTiling(mlir::OpPassManager &pm) {
    pm.addPass(createOptLoopTilingPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());

    auto &fpm = pm.nest<mlir::func::FuncOp>();
    fpm.addPass(mlir::affine::createAffineLoopInvariantCodeMotionPass());
    fpm.addPass(mlir::createCanonicalizerPass());
    fpm.addPass(mlir::createCSEPass());
}

void LoopParallelize(mlir::OpPassManager &pm) {
    pm.addPass(createOptLoopParallelizePass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
}

void Polyhedral(mlir::OpPassManager &pm) {
    auto &fpm = pm.nest<mlir::func::FuncOp>();
    fpm.addPass(mlir::affine::createAffineLoopNormalizePass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(createOptPolyhedralPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
}

}
