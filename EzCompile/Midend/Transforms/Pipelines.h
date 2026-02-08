//===-- Pipelines.h --------------------------------------------*- C++ -*-===//
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


#ifndef EZ_COMPILE_PIPELINES_H
#define EZ_COMPILE_PIPELINES_H

#include "llvm/Support/CommandLine.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Transforms/Passes.h"


namespace ezcompile {

/// Pipeline 选项
struct PipelineOptions
	: mlir::PassPipelineOptions<PipelineOptions> {

	Option<bool> enableCanonicalize{
		*this, "comp-base",
		llvm::cl::desc("Run canonicalize passes between lowering stages"),
		llvm::cl::init(false)};

};

void buildPipeline(mlir::OpPassManager &pm, const PipelineOptions &opt);

void registerPipelines();

std::unique_ptr<mlir::Pass> createLowerCompDimPass();
std::unique_ptr<mlir::Pass> createLowerCompFieldPass();
std::unique_ptr<mlir::Pass> createLowerCompPointsPass();
std::unique_ptr<mlir::Pass> createLowerCompApplyInitPass();

}

#endif //EZ_COMPILE_PIPELINES_H
