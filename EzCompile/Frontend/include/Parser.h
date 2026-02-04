//===-- Parser.h -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// 语法分析器：将 Token 序列解析为抽象语法树（AST）
//
//===----------------------------------------------------------------------===//


#ifndef EZ_COMPILE_PARSER_H
#define EZ_COMPILE_PARSER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/ADT/StringRef.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "Lexer.h"
#include "AST.h"
#include "DiagnosticBase.h"

namespace ezcompile {

/// 语法分析器：将 Token 序列解析为抽象语法树（AST）
///
/// Parser 使用递归下降和优先级爬升（precedence climbing）算法解析 comp 语言。
/// 在遇到错误时会尝试恢复（synchronization），以继续解析后续代码。
class Parser : public DiagnosticBase {
public:
    Parser(Lexer &lexer,
           llvm::SourceMgr &sourceMgr,
           int bufferID,
           mlir::MLIRContext *ctx);

    std::unique_ptr<ModuleAST> parseModule();

private:
    void advance();
    bool consume(Token::Kind k, llvm::StringRef msg);
    bool expect(Token::Kind k, llvm::StringRef msg) const;
    static llvm::SMLoc tokenEndLoc(const Token &t);

    /// 同步到下一个 section 关键字或 EOF
    void syncToSectionOrEOF();

    /// 同步到当前项的结束（分号或右花括号）
    void syncToItemEnd();

private:
    bool parseDeclarationsSection(ModuleAST &mod);
    bool parseEquationsSection(ModuleAST &mod);
    bool parseOptionsSection(ModuleAST &mod);

    /// 解析变量声明项：var_name[min, max, num]
    std::unique_ptr<VarDeclAST> parseVarDeclItem();

    /// 解析方程项：expr_lhs = expr_rhs;
    std::unique_ptr<EquationAST> parseEquationItem();

    /// 解析选项项：key: value;
    std::unique_ptr<OptionAST> parseOptionItem();

private:
    /// 解析表达式（优先级爬升算法）
    std::unique_ptr<ExprAST> parseExpr(int minPrec = 0);

    /// 解析一元表达式（+, -），检测连续运算符错误
    std::unique_ptr<ExprAST> parseUnary();

    /// 解析二元运算符的右侧（优先级爬升算法）
    std::unique_ptr<ExprAST> parseBinOpRHS(int minPrec,
                                           std::unique_ptr<ExprAST> lhs);

    /// 解析基本表达式（数字、字符串、标识符、函数调用、括号）
    std::unique_ptr<ExprAST> parsePrimary();

    /// 解析带符号的数字字面量（如 +6, -6）
    std::unique_ptr<ExprAST> parseSignedNumberLiteral(llvm::StringRef what);

    /// 获取当前 Token 的优先级：乘除(40) > 加减(20)
    int getTokPrecedence() const;

private:
    /// 解析选项字面量（只允许数字或字符串字面量）
    std::unique_ptr<ExprAST> parseOptionLiteral();

    /// 验证选项键值对的有效性
    void validateOption(const Token &keyTok, const ExprAST *value);

    /// 判断表达式是否为数字字面量（包括带符号的）
    static bool isNumberLiteral(const ExprAST *e) {
        if (llvm::isa<IntExprAST>(e)||llvm::isa<FloatExprAST>(e)) return true;
        if (auto now = llvm::dyn_cast<UnaryExprAST>(e)) {
            auto op = now->getOp();
            if ((op=='+'||op=='-')&&
                (llvm::isa<IntExprAST>(now->getOperand())||llvm::isa<FloatExprAST>(now->getOperand()))) {
                return true;
            }
        }
        return false;
    }

    static bool isStringLiteral(const ExprAST *e) { return llvm::isa<StringExprAST>(e); }

private:
    Lexer &lexer;           ///< 词法分析器引用

    Token curTok;           ///< 当前 Token

    Token prevTok;          ///< 前一个 Token，用于检测连续运算符
};


}

#endif //EZ_COMPILE_PARSER_H