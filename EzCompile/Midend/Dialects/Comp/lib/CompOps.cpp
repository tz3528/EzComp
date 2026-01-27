//===-- CompOps.cpp --------------------------------------------*- C++ -*-===//
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

#include "CompOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"

#define GET_OP_CLASSES
#include "CompOps.cpp.inc"

namespace ezcompile::comp {

mlir::LogicalResult ApplyInitOp::verify() { return mlir::success(); }
mlir::LogicalResult DimOp::verify() { return mlir::success(); }
mlir::LogicalResult DirichletOp::verify() { return mlir::success(); }
mlir::LogicalResult EnforceBoundaryOp::verify() { return mlir::success(); }
mlir::LogicalResult FieldOp::verify() { return mlir::success(); }
mlir::LogicalResult ForTimeOp::verify() { return mlir::success(); }
mlir::LogicalResult SampleOp::verify() { return mlir::success(); }
mlir::LogicalResult UpdateOp::verify() { return mlir::success(); }
mlir::LogicalResult CoordOp::verify() { return mlir::success(); }

}