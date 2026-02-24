//===-- Pipelines.cpp ------------------------------------------*- C++ -*-===//
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


#include "Pipelines.h"

namespace ezcompile {

void buildPipeline(mlir::OpPassManager &pm, const PipelineOptions &opt) {

	if (opt.enableLowerToBase.getValue() || opt.enableToLLVM.getValue()) {
		pm.addPass(createLowerCompCallPass());
		pm.addPass(createLowerCompPointsPass());
		pm.addPass(createLowerCompFieldPass());
		pm.addPass(createLowerCompApplyInitPass());
		pm.addPass(createLowerCompDirichletPass());
		pm.addPass(createLowerCompForTimePass());
		pm.addPass(createLowerCompUpdatePass());
		pm.addPass(createLowerCompDimPass());
		pm.addPass(createLowerCompSolvePass());
		pm.addPass(createLowerCompProblemPass());

		// 优化：规范化 + 常量折叠 + CSE
		pm.addPass(mlir::createCanonicalizerPass());
		pm.addPass(mlir::createCSEPass());
	}

	//===--------------------------------------------------------------------===//
	// 阶段2：Affine → SCF
	//===--------------------------------------------------------------------===//

	if (opt.enableAffineToSCF.getValue() || opt.enableToLLVM.getValue()) {
		pm.addPass(mlir::createLowerAffinePass());
		pm.addPass(mlir::createCanonicalizerPass());
	}

	//===--------------------------------------------------------------------===//
	// 阶段3：SCF → ControlFlow
	//===--------------------------------------------------------------------===//

	if (opt.enableSCFToCF.getValue() || opt.enableToLLVM.getValue()) {
		pm.addPass(mlir::createSCFToControlFlowPass());
		pm.addPass(mlir::createCanonicalizerPass());
	}

	//===--------------------------------------------------------------------===//
	// 阶段4：基础方言 → LLVM
	//===--------------------------------------------------------------------===//

	if (opt.enableToLLVM.getValue()) {
		pm.addPass(mlir::createConvertToLLVMPass());
		pm.addPass(mlir::createCanonicalizerPass());
	}

}

void registerPipelines() {
	mlir::PassPipelineRegistration<PipelineOptions>(
		"lowering",
		"Lower Comp dialect via staged lowering with configurable options",
		buildPipeline);
}

} // namespace ezcompile