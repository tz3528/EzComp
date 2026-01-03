//===-- OptionTable.cpp ----------------------------------------*- C++ -*-===//
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

#include "llvm/ADT/StringRef.h"

#include "OptionTable.h"

namespace ezcompile {

using llvm::StringRef;

// ---------------- Construction ----------------
mlir::LogicalResult OptionsTable::registerAllOptions(OptionsTable &opts) {
#define OPT_STR(nameIdent, defLit, validatorExpr)                                   \
    do {                                                                            \
        OptionsTable::Spec sp;                                                      \
        sp.kind = OptionsTable::Kind::Str;                                          \
        sp.value = defLit;                                                          \
        sp.check = OptionsTable::validatorExpr;                                     \
        if (failed(opts.registerOption(#nameIdent, sp))) {               \
            return mlir::failure();                                                 \
        }                                                                           \
    } while (0);

#define OPT_INT(nameIdent, defVal, validatorExpr)                                   \
    do {                                                                            \
        OptionsTable::Spec sp;                                                      \
        sp.kind = OptionsTable::Kind::IntI64;                                       \
        sp.value = defVal;                                                          \
        sp.check = OptionsTable::validatorExpr;                                     \
        if (failed(opts.registerOption(#nameIdent, sp))) {               \
            return mlir::failure();                                                 \
        }                                                                           \
    } while (0);

#define OPT_F64(nameIdent, defVal, validatorExpr)                                   \
    do {                                                                            \
        OptionsTable::Spec sp;                                                      \
        sp.kind = OptionsTable::Kind::FloatF64;                                     \
        sp.value = defVal;                                                          \
        sp.check = OptionsTable::validatorExpr;                                     \
        if (failed(opts.registerOption(#nameIdent, sp))) {               \
            return mlir::failure();                                                 \
        }                                                                           \
    } while (0);

#include "Options.def"

#undef OPT_STR
#undef OPT_INT
#undef OPT_F64

    return mlir::success();
}

llvm::Expected<OptionsTable> OptionsTable::createWithDefaults() {
    OptionsTable t;

    if (llvm::failed(registerAllOptions(t))) {
        return llvm::make_error<llvm::StringError>("Operation failed", llvm::inconvertibleErrorCode());
    }

    return t;
}

mlir::LogicalResult OptionsTable::registerOption(std::string name, Spec &spec) {
    // 避免重复 key
    if (specs_.count(name)) {
        llvm::errs() << "Duplicate option name: " + name;
        return mlir::failure();
    }

    //值类型必须匹配 kind
    if (kindOf(spec.value) != spec.kind) {
        llvm::errs() << "Default kind mismatch for " + name + ": expect " + kindName(spec.kind) +
            ", got " + valueTypeName(spec.value);
        return mlir::failure();
    }

    // 默认值也跑一遍 validator：能更早发现 options.def 写错（例如范围不对）
    if (spec.check) {
        if (auto e = spec.check(spec.value)) {
            llvm::errs() << "Default value invalid for " + name + ": " + llvm::toString(std::move(e));
            return mlir::failure();
        }
    }
    else {
        llvm::errs() << "No Checker " + name;
        return mlir::failure();
    }

    specs_[name] = std::move(spec);
    return mlir::success();
}

mlir::LogicalResult OptionsTable::set(std::string name, Value v, std::string &err) {
    if (specs_.find(name) == specs_.end()) {
        err = "Option '" + name + "' is not registered";
        return mlir::failure();
    }
    if (!sameKind(specs_[name].value, v)) {
        err = "Option '" + name + "' is " + valueTypeName(specs_[name].value) + " type. But v is "+
            valueTypeName(v) + "type";
        return mlir::failure();
    }
    if (auto ans = specs_[name].check(v)) {
        err = llvm::toString(std::move(ans));
        return mlir::failure();
    }

    specs_[name].value = std::move(v);
    return mlir::success();
}

mlir::FailureOr<std::string> OptionsTable::getStr(std::string name, std::string &err) {
    if (specs_.find(name) == specs_.end()) {
        err = "Option '" + name + "' is not registered";
        return mlir::failure();
    }
    if (specs_[name].kind != Kind::Str) {
        err = "Option '" + name + "' is " + valueTypeName(specs_[name].value);
        return mlir::failure();
    }
    return std::get<std::string>(specs_[name].value);
}

mlir::FailureOr<int64_t> OptionsTable::getInt(std::string name, std::string &err) {
    if (specs_.find(name) == specs_.end()) {
        err = "Option '" + name + "' is not registered";
        return mlir::failure();
    }
    if (specs_[name].kind != Kind::Str) {
        err = "Option '" + name + "' is " + valueTypeName(specs_[name].value);
        return mlir::failure();
    }
    return std::get<int64_t>(specs_[name].value);
}

mlir::FailureOr<double> OptionsTable::getFloat(std::string name, std::string &err) {
    if (specs_.find(name) == specs_.end()) {
        err = "Option '" + name + "' is not registered";
        return mlir::failure();
    }
    if (specs_[name].kind != Kind::Str) {
        err = "Option '" + name + "' is " + valueTypeName(specs_[name].value);
        return mlir::failure();
    }
    return std::get<double>(specs_[name].value);
}

// ---------------- Validators ----------------

OptionsTable::CheckFn OptionsTable::any() {
    return [](const Value &v) -> llvm::Error {
        return llvm::Error::success();
    };
}

OptionsTable::CheckFn OptionsTable::oneOfStrings(std::initializer_list<std::string> allowed) {
    std::vector<std::string> allowedVec(allowed.begin(), allowed.end());
    return [allowedVec = std::move(allowedVec)](const Value &v) -> llvm::Error {
        const auto &s = std::get<std::string>(v);

        for (const auto &a : allowedVec) {
            if (a == s) return llvm::Error::success();
        }

        std::string msg = "must be one of: ";
        bool first = true;
        for (const auto &a : allowedVec) {
            if (!first) msg += ", ";
            first = false;
            msg += a;
        }

        return llvm::createStringError(llvm::inconvertibleErrorCode(), "%s", msg.c_str());
    };
}

OptionsTable::CheckFn OptionsTable::oneOfInt64(std::initializer_list<int64_t> allowed) {
    std::vector<int64_t> allowedVec(allowed.begin(), allowed.end());
    return [allowedVec = std::move(allowedVec)](const Value &v) -> llvm::Error {
        const auto &x = std::get<int64_t>(v);

        for (auto a : allowedVec) {
            if (a == x) return llvm::Error::success();
        }

        std::string msg = "must be one of: ";
        bool first = true;
        for (auto a : allowedVec) {
            if (!first) msg += ", ";
            first = false;
            msg += std::to_string(a);
        }

        return llvm::createStringError(
            llvm::inconvertibleErrorCode(), "%s", msg.c_str());
    };
}

OptionsTable::CheckFn OptionsTable::oneOfFloat(std::initializer_list<double> allowed) {
    std::vector<double> allowedVec(allowed.begin(), allowed.end());
    return [allowedVec = std::move(allowedVec)](const Value &v) -> llvm::Error {
        const auto &x = std::get<double>(v);

        for (auto a : allowedVec) {
            if (a == x) return llvm::Error::success();
        }

        std::string msg = "must be one of: ";
        bool first = true;
        for (auto a : allowedVec) {
            if (!first) msg += ", ";
            first = false;
            msg += std::to_string(a);
        }

        return llvm::createStringError(
            llvm::inconvertibleErrorCode(), "%s", msg.c_str());
    };
}

// ---------------- Kind helpers ----------------

OptionsTable::Kind OptionsTable::kindOf(const Value &v) {
    if (std::holds_alternative<std::string>(v)) return Kind::Str;
    if (std::holds_alternative<int64_t>(v)) return Kind::IntI64;
    return Kind::FloatF64;
}

std::string OptionsTable::kindName(Kind k) {
    switch (k) {
    case Kind::Str:
        return "string";
    case Kind::IntI64:
        return "int64";
    case Kind::FloatF64:
        return "double";
    default:
        return "unknown";
    }
}

std::string OptionsTable::valueTypeName(const Value &v) {
    return std::visit([](const auto &x) -> std::string {
        using T = std::decay_t<decltype(x)>;

        if constexpr (std::is_same_v<T, std::string>)
            return "string";
        else if constexpr (std::is_same_v<T, int64_t>)
            return "int64";
        else if constexpr (std::is_same_v<T, double>)
            return "double";
        else
            return "unknown";
    }, v);
}

bool OptionsTable::sameKind(const Value &a, const Value &b) {
    return kindOf(a) == kindOf(b);
}

}