//===-- OptPipelines.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pass 管线定义
// 定义优化管线配置
//
//===----------------------------------------------------------------------===//


#ifndef EZ_RESEARCH_OPT_PIPELINES_H
#define EZ_RESEARCH_OPT_PIPELINES_H

#include "llvm/Support/CommandLine.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Transforms/Passes.h"

namespace ezresearch {

struct OptimizationOptions : mlir::PassPipelineOptions<OptimizationOptions> {
    Option<bool> enableHoistBoundary{
        *this, "hoist-boundary",
        llvm::cl::desc("Hoest boundary equation to the outside of time loop"),
        llvm::cl::init(false)};

};

std::unique_ptr<mlir::Pass> createOptBoundaryHoistPass();

void HoistBoundary(mlir::OpPassManager &pm);

}

#endif //EZ_RESEARCH_OPT_PIPELINES_H