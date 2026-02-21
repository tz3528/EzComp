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

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Transforms/Passes.h"


namespace ezcompile {

/// Pipeline 选项
struct PipelineOptions : mlir::PassPipelineOptions<PipelineOptions> {

	Option<bool> enableLowerToBase{
		*this, "comp-base",
		llvm::cl::desc("Run canonicalize passes between lowering stages"),
		llvm::cl::init(false)};

	Option<bool> enableAffineToSCF{
		*this, "affine-scf",
		llvm::cl::desc("Lower Affine dialect to SCF dialect"),
		llvm::cl::init(false)};

	Option<bool> enableSCFToCF{
		*this, "scf-cf",
		llvm::cl::desc("Lower SCF dialect to ControlFlow dialect"),
		llvm::cl::init(false)};

	Option<bool> enableToLLVM{
		*this, "base-llvm",
		llvm::cl::desc("Lower Math, Arith, MemRef, and CF to LLVM dialect"),
		llvm::cl::init(false)};


};

void buildPipeline(mlir::OpPassManager &pm, const PipelineOptions &opt);

void registerPipelines();

std::unique_ptr<mlir::Pass> createLowerCompDimPass();
std::unique_ptr<mlir::Pass> createLowerCompFieldPass();
std::unique_ptr<mlir::Pass> createLowerCompPointsPass();
std::unique_ptr<mlir::Pass> createLowerCompApplyInitPass();
std::unique_ptr<mlir::Pass> createLowerCompForTimePass();
std::unique_ptr<mlir::Pass> createLowerCompDirichletPass();
std::unique_ptr<mlir::Pass> createLowerCompUpdatePass();
std::unique_ptr<mlir::Pass> createLowerCompSolvePass();
std::unique_ptr<mlir::Pass> createLowerCompProblemPass();
std::unique_ptr<mlir::Pass> createLowerCompCallPass();
std::unique_ptr<mlir::Pass> createCleanupUnrealizedCastPass();

}

#endif //EZ_COMPILE_PIPELINES_H
