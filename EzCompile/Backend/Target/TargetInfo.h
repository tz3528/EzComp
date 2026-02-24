//===-- TargetInfo.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// 目标平台信息查询接口
//
//===----------------------------------------------------------------------===//


#ifndef EZ_COMPILE_TARGET_INFO_H
#define EZ_COMPILE_TARGET_INFO_H

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Triple.h"

#include <string>

namespace ezcompile::target {

/// 目标平台信息封装
/// 提供目标架构相关的查询接口
class TargetInfo {
public:
    explicit TargetInfo(llvm::TargetMachine &targetMachine);

    //===------------------------------------------------------------------===//
    // 基础信息
    //===------------------------------------------------------------------===//

    /// 获取目标三元组
    llvm::Triple getTriple() const;

    /// 获取目标三元组字符串
    std::string getTripleString() const;

    /// 获取目标 CPU 名称
    llvm::StringRef getCPUName() const;

    /// 获取目标特性字符串
    llvm::StringRef getTargetFeatures() const;

    //===------------------------------------------------------------------===//
    // 数据布局信息
    //===------------------------------------------------------------------===//

    /// 获取数据布局
    const llvm::DataLayout &getDataLayout() const;

    /// 获取指针宽度（位数）
    unsigned getPointerWidth() const;

    /// 获取指针大小（字节数）
    unsigned getPointerSize() const;

    /// 获取类型的大小（字节数）
    unsigned getTypeSize(llvm::Type *type) const;

    /// 获取类型的对齐要求
    unsigned getTypeAlignment(llvm::Type *type) const;

    //===------------------------------------------------------------------===//
    // 架构特性
    //===------------------------------------------------------------------===//

    /// 是否为 64 位架构
    bool is64Bit() const;

    /// 是否为小端序
    bool isLittleEndian() const;

    /// 获取架构名称 (x86_64, aarch64, etc.)
    std::string getArchName() const;

    /// 获取厂商名称 (unknown, pc, apple, etc.)
    std::string getVendorName() const;

    /// 获取操作系统名称 (linux, windows, etc.)
    std::string getOSName() const;

private:
    llvm::TargetMachine &targetMachine;
    llvm::DataLayout dataLayout;
};

} // namespace ezcompile::target

#endif // EZ_COMPILE_TARGET_INFO_H
