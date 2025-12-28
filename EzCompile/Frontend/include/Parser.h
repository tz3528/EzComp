//===-- Parser.h -----------------------------------------------*- C++ -*-===//
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

    void syncToSectionOrEOF();
    void syncToItemEnd();

private:
    bool parseDeclarationsSection(ModuleAST &mod);
    bool parseEquationsSection(ModuleAST &mod);
    bool parseOptionsSection(ModuleAST &mod);

    std::unique_ptr<VarDeclAST> parseVarDeclItem();
    std::unique_ptr<EquationAST> parseEquationItem();
    std::unique_ptr<OptionAST> parseOptionItem();

private:
    std::unique_ptr<ExprAST> parseExpr(int minPrec = 0);
    std::unique_ptr<ExprAST> parseUnary();
    std::unique_ptr<ExprAST> parseBinOpRHS(int minPrec,std::unique_ptr<ExprAST> lhs);
    std::unique_ptr<ExprAST> parsePrimary();
    std::unique_ptr<ExprAST> parseSignedNumberLiteral(llvm::StringRef what);
    int getTokPrecedence() const;

private:
    // options 的 value 只允许：数字字面量(可带前导 +/-) 或字符串字面量
    std::unique_ptr<ExprAST> parseOptionLiteral();
    void validateOption(const Token &keyTok, const ExprAST *value);

    static bool isNumberLiteral(const ExprAST *e) {
        if (llvm::isa<NumberExprAST>(e)) return true;
        if (auto now = llvm::dyn_cast<UnaryExprAST>(e)) {
            auto op = now->getOp();
            if ((op=='+'||op=='-')&&llvm::isa<NumberExprAST>(now->getOperand())) {
                return true;
            }
        }
        return false;
    }
    static bool isStringLiteral(const ExprAST *e) { return llvm::isa<StringExprAST>(e); }

private:
    Lexer &lexer;

    Token curTok;

};


}

#endif //EZ_COMPILE_PARSER_H
