//===-- LowerPipelines.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pass 管线定义
// 定义编译流程的降级和优化管线配置
//
//===----------------------------------------------------------------------===//


#ifndef EZ_COMPILE_LOWER_PIPELINES_H
#define EZ_COMPILE_LOWER_PIPELINES_H

#include "llvm/Support/CommandLine.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Transforms/Passes.h"


namespace ezcompile {

/// Pipeline 选项，控制四阶段降级流程
struct LoweringOptions : mlir::PassPipelineOptions<LoweringOptions> {

	/// Comp 方言 → 基础方言 (Affine/Arith/MemRef)
	Option<bool> enableLowerToBase{
		*this, "comp-base",
		llvm::cl::desc("Lower Comp dialect to base dialects"),
		llvm::cl::init(false)};

	/// Affine → SCF
	Option<bool> enableAffineToSCF{
		*this, "affine-scf",
		llvm::cl::desc("Lower Affine dialect to SCF dialect"),
		llvm::cl::init(false)};

	/// SCF → ControlFlow
	Option<bool> enableSCFToCF{
		*this, "scf-cf",
		llvm::cl::desc("Lower SCF dialect to ControlFlow dialect"),
		llvm::cl::init(false)};

	/// 基础方言 → LLVM 方言
	Option<bool> enableToLLVM{
		*this, "base-llvm",
		llvm::cl::desc("Lower base dialects to LLVM dialect"),
		llvm::cl::init(false)};

};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

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
std::unique_ptr<mlir::Pass> createLowerCompDeltaPass();

void LowerToBase(mlir::OpPassManager &pm);
void AffineToSCF(mlir::OpPassManager &pm);
void SCFToCF(mlir::OpPassManager &pm);
void ToLLVM(mlir::OpPassManager &pm);

} // namespace ezcompile

#endif //EZ_COMPILE_LOWER_PIPELINES_H