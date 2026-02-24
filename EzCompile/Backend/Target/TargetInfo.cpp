//===-- TargetInfo.cpp -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// 目标平台信息查询实现
//
//===----------------------------------------------------------------------===//


#include "TargetInfo.h"

#include "llvm/IR/Type.h"

namespace ezcompile::target {

TargetInfo::TargetInfo(llvm::TargetMachine &targetMachine)
    : targetMachine(targetMachine),
      dataLayout(targetMachine.createDataLayout()) {}

//===----------------------------------------------------------------------===//
// 基础信息
//===----------------------------------------------------------------------===//

llvm::Triple TargetInfo::getTriple() const {
    return targetMachine.getTargetTriple();
}

std::string TargetInfo::getTripleString() const {
    return getTriple().str();
}

llvm::StringRef TargetInfo::getCPUName() const {
    return targetMachine.getTargetCPU();
}

llvm::StringRef TargetInfo::getTargetFeatures() const {
    return targetMachine.getTargetFeatureString();
}

//===----------------------------------------------------------------------===//
// 数据布局信息
//===----------------------------------------------------------------------===//

const llvm::DataLayout &TargetInfo::getDataLayout() const {
    return dataLayout;
}

unsigned TargetInfo::getPointerWidth() const {
    return dataLayout.getPointerSizeInBits();
}

unsigned TargetInfo::getPointerSize() const {
    return dataLayout.getPointerSize();
}

unsigned TargetInfo::getTypeSize(llvm::Type *type) const {
    return dataLayout.getTypeAllocSize(type);
}

unsigned TargetInfo::getTypeAlignment(llvm::Type *type) const {
    return dataLayout.getABITypeAlign(type).value();
}

//===----------------------------------------------------------------------===//
// 架构特性
//===----------------------------------------------------------------------===//

bool TargetInfo::is64Bit() const {
    return getPointerWidth() == 64;
}

bool TargetInfo::isLittleEndian() const {
    return dataLayout.isLittleEndian();
}

std::string TargetInfo::getArchName() const {
    return getTriple().getArchName().str();
}

std::string TargetInfo::getVendorName() const {
    return getTriple().getVendorName().str();
}

std::string TargetInfo::getOSName() const {
    return getTriple().getOSName().str();
}

} // namespace ezcompile::target
