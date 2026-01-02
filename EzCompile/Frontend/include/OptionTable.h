//===-- OptionTable.h ------------------------------------------*- C++ -*-===//
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


#ifndef OPTIONTABLE_H
#define OPTIONTABLE_H

#include "mlir/Support/LogicalResult.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"

#include <functional>
#include <string>
#include <variant>

namespace ezcompile {

class OptionsTable {
public:
    enum class Kind { Str, IntI64, FloatF64 };

    // 运行时值：仍然只支持三类
    using Value = std::variant<std::string, int64_t, double>;

    // validator：成功返回 llvm::Error::success()，失败返回带消息的 Error
    using CheckFn = std::function<llvm::Error(const Value &)>;

    struct Spec {
        Kind kind;
        Value value;
        CheckFn check;            // 允许为空：仅做类型检查
    };

    // --- 创建：注册全部 spec 并初始化为默认值（方案B：编译期 include options.def）---
    static llvm::Expected<OptionsTable> createWithDefaults();
    static mlir::LogicalResult registerAllOptions(OptionsTable &opts);
    mlir::LogicalResult registerOption(std::string name, Spec &spec);

    // --- 查询/重置 ---
    bool has(std::string name) const;

    // --- set（写入 + 校验）---
    mlir::LogicalResult set(std::string name, Value v, std::string &err);

    // --- get（带错误信息的安全读取；返回的是“值拷贝”）---
    mlir::FailureOr<std::string> getStr(std::string name, std::string &err);
    mlir::FailureOr<int64_t> getInt(std::string name, std::string &err);
    mlir::FailureOr<double> getFloat(std::string name, std::string &err);

public:
    // --- 校验器工厂（写在 options.def 的 validatorExpr 里）---
    static CheckFn any();  // 不额外检查（只检查类型）
    static CheckFn oneOfStrings(std::initializer_list<std::string> allowed);
    static CheckFn oneOfInt64(std::initializer_list<int64_t> allowed);
    static CheckFn oneOfFloat(std::initializer_list<double> allowed);

public:

    static Kind kindOf(const Value &v);
    static std::string kindName(Kind k);
    static std::string valueTypeName(const Value &v);
    static bool sameKind(const Value &a, const Value &b);

private:
    //用于记录所有的选项
    llvm::StringMap<Spec> specs_;

    //记录待求函数的信息
    struct FunctionSig {
        std::string name;                   //函数名
        std::vector<std::string> args;      //函数参数
        std::string text;                   //对应的文本
    } targetFunc;
};

}

#endif //OPTIONTABLE_H
