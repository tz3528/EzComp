//===-- Lexer.h ------------------------------------------------*- C++ -*-===//
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


#ifndef EZ_COMPILE_LEXER_H
#define EZ_COMPILE_LEXER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"

namespace mlir {
class MLIRContext;
} // namespace mlir

namespace ezcompile {

/// Token 只做“切片”：
/// - kind：类别（标识符/数字/符号/结构关键字等）
/// - loc：起始位置（SMLoc 指向 buffer 内某个字符）
/// - spelling：源码中的原始文本切片（StringRef 指向同一份 buffer）
///
/// 设计要点：
/// 1) 不分配内存、不拷贝字符串：spelling 直接引用 SourceMgr 的 buffer，
///    这符合 LLVM/MLIR 的“轻量前端”风格。
/// 2) 词法层不做语义判断：比如一元负号、diff 是否内建函数，都交给 parser/语义层。
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

class Lexer {
public:
    /// bufferID 是 SourceMgr 中 MemoryBuffer 的 index（通常 0）。
    /// context 可为空：为空则退化为 SourceMgr 自带报错；不为空则走 MLIR 诊断系统。
    Lexer(llvm::SourceMgr &sourceMgr,
          int bufferID,
          mlir::MLIRContext *context);

    /// 读取下一个 token。
    Token lex();

    bool hadError() const { return sawError; }

private:
    /// 跳过空白与注释。
    /// 注释语法（comp 源码）：
    /// - # line comment  （从 # 到行尾）
    ///
    /// 说明：
    /// 1) 词法层只负责跳过注释，不产出注释 token。
    /// 2) 这样保持 lexer 无状态、无上下文，便于后续 parser/IRGen 统一处理。
    void skipWhitespaceAndComments();

    /// 识别 identifier 或结构关键字（declarations/equations/options）。
    /// 注意：diff/sin/exp/log 等一律作为 identifier，
    /// 后续由 parser 看到 "identifier '(' ... ')'" 来形成 CallExpr。
    Token lexIdentifierOrKeyword();

    /// 识别 number（不吃前导 +/-，避免把 “-6” 粘成一个 token，保持无上下文）。
    Token lexNumber();

    Token lexStringLiteral();

    /// diagnostics：尽量走 MLIR 的 emitError（带 FileLineColLoc），
    /// 这样后续 parser/IRGen 也能复用同一套报错风格。
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
