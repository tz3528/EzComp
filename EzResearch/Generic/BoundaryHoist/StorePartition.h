//===-- StorePartition.h -------------------------------------- -*- C++ -*-===//
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


#ifndef EZ_RESEARCH_BOUNDARYHOIST_STOREPARTITION_H
#define EZ_RESEARCH_BOUNDARYHOIST_STOREPARTITION_H

#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include <vector>

#include "Utils/QueryUtil.h"

namespace ezresearch {

class StorePartition {
public:
    StorePartition(mlir::memref::AllocOp alloc) : alloc(alloc) {}

    void emplace(mlir::memref::StoreOp store);

    std::vector<mlir::memref::StoreOp> analyze();
private:
    bool proveNoOverlap1D(mlir::Value a, mlir::Value b);

    bool proveNoSpatialOverlap(mlir::memref::StoreOp boundary, mlir::memref::StoreOp iter);

    mlir::memref::AllocOp alloc;

    std::vector<mlir::memref::StoreOp> iter_store;
    std::vector<mlir::memref::StoreOp> boundary_store;
};

}

#endif //EZ_RESEARCH_BOUNDARYHOIST_STOREPARTITION_H