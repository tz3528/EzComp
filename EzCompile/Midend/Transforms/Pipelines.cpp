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
	// 1.将所有应当是常量属性的降级，分别有dim、coord和points
	pm.addPass(createLowerCompPointsToArithPass());

	// 2.实例化所有的alloc、store、load

	// 3.消解循环

	// 4.将所有的affine、memref和arith降级为llvm方言
}

void registerPipelines() {
	mlir::PassPipelineRegistration<PipelineOptions>(
		"lowering",
		"Lower MyProject dialect via staged lowering with configurable options",
		buildPipeline);
}// set arg ../Examples/EcCompile/Midend/basic/2d_heat_equation.comp  -emit=mlir --pass-pipeline="builtin.module(lowering{comp-base=true})"

}
