//===-- Lexer.h ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// 词法分析器：将 comp 源代码分解为 Token 序列
//
//===----------------------------------------------------------------------===//


#ifndef EZ_COMPILE_LEXER_H
#define EZ_COMPILE_LEXER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"

namespace mlir {
class MLIRContext;
} // namespace mlir

namespace ezcompile {

/// 词法单元，表示源代码中的一个最小语义单位
///
/// Token 采用"切片"设计：不分配内存、不拷贝字符串，直接引用 SourceMgr 的 buffer。
/// 词法层不做语义判断，语义由后续的 Parser 和语义分析处理。
class Token {
public:
    enum Kind : uint8_t {
        eof,
        error,

        identifier, // [A-Za-z_][A-Za-z0-9_]*
        number,     // 123 / 3.14 / 1e-6 之类（负号不并入 number）
        string,

        // 关键字
        kw_declarations,
        kw_equations,
        kw_options,

        // 符号
        l_brace,    // {
        r_brace,    // }
        l_paren,    // (
        r_paren,    // )
        l_square,   // [
        r_square,   // ]
        semicolon,  // ;
        comma,      // ,
        colon,      // :
        equal,      // =
        plus,       // +
        minus,      // -
        star,       // *
        slash       // /
    };

    Token() = default;
    Token(Kind kind, llvm::SMLoc loc, llvm::StringRef spelling)
        : kind(kind), loc(loc), spelling(spelling) {}

    Kind getKind() const { return kind; }
    llvm::SMLoc getLoc() const { return loc; }
    llvm::StringRef getSpelling() const { return spelling; }
    bool is(Kind k) const { return kind == k; }
    bool isNot(Kind k) const { return kind != k; }

private:
    Kind kind = eof;
    llvm::SMLoc loc;
    llvm::StringRef spelling;
};

/// 词法分析器：将源代码分解为 Token 序列
class Lexer {
public:
    Lexer(llvm::SourceMgr &sourceMgr,
          int bufferID,
          mlir::MLIRContext *context);

    Token lex();
    bool hadError() const { return sawError; }

private:
    /// 跳过空白字符与注释（# line comment）
    void skipWhitespaceAndComments();

    /// 识别标识符或结构关键字（declarations/equations/options）
    Token lexIdentifierOrKeyword();

    /// 识别数字字面量：123, 3.14, .5, 1e-6（不处理前导 +/-）
    Token lexNumber();

    /// 识别字符串字面量（支持转义字符如 \"）
    Token lexStringLiteral();

    void emitError(llvm::SMLoc loc, llvm::StringRef message);

private:
    llvm::SourceMgr &sourceMgr;
    const int bufferID;
    mlir::MLIRContext *context = nullptr;

    const char *bufferStart = nullptr;
    const char *bufferEnd = nullptr;
    const char *curPtr = nullptr;

    bool sawError = false;
};

} // namespace ezcompile

#endif // EZ_COMPILE_LEXER_H