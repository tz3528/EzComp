//===-- BakendDriver.h -----------------------------------------*- C++ -*-===//
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


#ifndef EZ_COMPILE_BAKEND_DRIVER_H
#define EZ_COMPILE_BAKEND_DRIVER_H

#include "mlir/Support/LogicalResult.h"

namespace ezcompile {

class Bakend {
public:


	mlir::LogicalResult run();

private:

};

}

#endif //EZ_COMPILE_BAKEND_DRIVER_H
