//===-- Pipelines.cpp ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// pass管线，用于管理降级和优化的顺序
//
//===----------------------------------------------------------------------===//


#include "Pipelines.h"

namespace ezcompile {

void buildPipeline(mlir::OpPassManager &pm, const PipelineOptions &opt) {

	pm.addPass(createLowerCompPointsPass());
	pm.addPass(createLowerCompFieldPass());
	pm.addPass(createLowerCompApplyInitPass());
	pm.addPass(createLowerCompDirichletPass());
	pm.addPass(createLowerCompForTimePass());
	pm.addPass(createLowerCompUpdatePass());
	pm.addPass(createLowerCompDimPass());

}

void registerPipelines() {
	mlir::PassPipelineRegistration<PipelineOptions>(
		"lowering",
		"Lower MyProject dialect via staged lowering with configurable options",
		buildPipeline);
}// set arg ../Examples/EcCompile/Midend/basic/2d_heat_equation.comp  -emit=mlir --pass-pipeline="lowering{comp-base=true}"

}
