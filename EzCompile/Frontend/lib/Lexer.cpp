//===-- Lexer.cpp ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// 词法分析器实现
//
//===----------------------------------------------------------------------===//


#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/ADT/StringExtras.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LLVM.h"

#include "Lexer.h"

namespace ezcompile {

using llvm::StringRef;

/// 判断字符是否为标识符起始字符（字母或下划线）
static bool isIdentStart(char c) {
    return llvm::isAlpha(c) || c == '_';
}

/// 判断字符是否为标识符主体字符（字母、数字或下划线）
static bool isIdentBody(char c) {
    return llvm::isAlnum(c) || c == '_';
}

Lexer::Lexer(llvm::SourceMgr &sourceMgr,
             int bufferID,
             mlir::MLIRContext *context)
    : sourceMgr(sourceMgr),
      bufferID(bufferID),
      context(context) {
    /// 从 SourceMgr 获取 buffer，并用 begin/end 作为指针游标
    ///
    /// 这样保证：
    /// - lex() 全程是线性扫描
    /// - token 的 spelling 只是 StringRef（指向原 buffer），不分配内存
    const llvm::MemoryBuffer *buf = sourceMgr.getMemoryBuffer(bufferID);
    StringRef s = buf->getBuffer();
    bufferStart = s.begin();
    bufferEnd = s.end();
    curPtr = bufferStart;
}

Token Lexer::lex() {
    /// 先跳过空白/注释，保证调用者看到的都是有效 token
    skipWhitespaceAndComments();

    llvm::SMLoc loc = llvm::SMLoc::getFromPointer(curPtr);
    if (curPtr >= bufferEnd)
        return Token(Token::eof, loc, StringRef());

    const char *tokStart = curPtr;
    char c = *curPtr;

    /// identifier / keyword（结构关键字在这里特判）
    if (isIdentStart(c))
        return lexIdentifierOrKeyword();

    /// number：
    /// - 允许 "123"、"3.14"、".5"、"1e-6"
    /// - 但不把 '-' 并入 number（保持无上下文），否则 options 里的 -6 会粘成一个 token，
    ///   解析器处理一元负号会更别扭
    if (llvm::isDigit(c) ||
        (c == '.' && (curPtr + 1) < bufferEnd && llvm::isDigit(curPtr[1])))
        return lexNumber();

    if (c == '"')
        return lexStringLiteral();

    /// 单字符符号：直接返回，不做更复杂的合并（目前语法也不需要）
    ++curPtr;
    switch (c) {
    case '{': return Token(Token::l_brace, loc, "{");
    case '}': return Token(Token::r_brace, loc, "}");
    case '(': return Token(Token::l_paren, loc, "(");
    case ')': return Token(Token::r_paren, loc, ")");
    case '[': return Token(Token::l_square, loc, "[");
    case ']': return Token(Token::r_square, loc, "]");
    case ';': return Token(Token::semicolon, loc, ";");
    case ',': return Token(Token::comma, loc, ",");
    case ':': return Token(Token::colon, loc, ":");
    case '=': return Token(Token::equal, loc, "=");
    case '+': return Token(Token::plus, loc, "+");
    case '-': return Token(Token::minus, loc, "-");
    case '*': return Token(Token::star, loc, "*");
    case '/': return Token(Token::slash, loc, "/");
    default:
        /// 词法错误：未知字符
        emitError(loc, "unexpected character");
        return Token(Token::error, loc, StringRef(tokStart, 1));
    }
}

void Lexer::skipWhitespaceAndComments() {
    while (curPtr < bufferEnd) {
        /// 1) 空白：包括空格、tab、换行等，直接跳过
        if (llvm::isSpace(*curPtr)) {
            ++curPtr;
            continue;
        }

        /// 2) 行注释：# ... \n
        ///    - 无论 '#' 出现在行首还是代码后（如 ";#comment"），都能被下一次 lex() 入口吃掉
        if (*curPtr == '#') {
            ++curPtr; // 吃掉 '#'
            while (curPtr < bufferEnd && *curPtr != '\n')
                ++curPtr;
            continue;
        }

        /// 既不是空白也不是注释：停止跳过，回到 lex() 处理 token
        break;
    }
}

Token Lexer::lexIdentifierOrKeyword() {
    const char *tokStart = curPtr;
    ++curPtr;
    while (curPtr < bufferEnd && isIdentBody(*curPtr))
        ++curPtr;

    StringRef spelling(tokStart, curPtr - tokStart);

    /// 只把"语法结构保留字"当 keyword
    /// diff/sin/exp/log 等全部是 identifier，后续由 parser 识别函数调用
    Token::Kind kind =
        llvm::StringSwitch<Token::Kind>(spelling)
            .Case("declarations", Token::kw_declarations)
            .Case("equations", Token::kw_equations)
            .Case("options", Token::kw_options)
            .Default(Token::identifier);

    return Token(kind,
                 llvm::SMLoc::getFromPointer(tokStart),
                 spelling);
}

Token Lexer::lexNumber() {
    const char *tokStart = curPtr;

    /// 允许以 '.' 开头的浮点（如 .5）
    if (*curPtr == '.')
        ++curPtr;

    /// 整数部分（或者 . 后面的第一段 digits）
    while (curPtr < bufferEnd && llvm::isDigit(*curPtr))
        ++curPtr;

    /// 小数部分：digits '.' digits
    /// 注意：如果写成 "123." 且后面不是 digit，这里不把 '.' 吃进 number，
    /// 让 parser 决定这是否合法（目前你的语言未必需要 "123." 这种写法）
    if (curPtr < bufferEnd &&
        *curPtr == '.' &&
        (curPtr + 1) < bufferEnd &&
        llvm::isDigit(curPtr[1])) {
        ++curPtr;
        while (curPtr < bufferEnd && llvm::isDigit(*curPtr))
            ++curPtr;
    }

    /// 科学计数：e/E [+-]? digits
    if (curPtr < bufferEnd &&
        (*curPtr == 'e' || *curPtr == 'E')) {
        const char *expStart = curPtr;
        ++curPtr;
        if (curPtr < bufferEnd &&
            (*curPtr == '+' || *curPtr == '-'))
            ++curPtr;

        /// e 后必须跟至少一个 digit，否则报错
        if (curPtr >= bufferEnd || !llvm::isDigit(*curPtr)) {
            emitError(llvm::SMLoc::getFromPointer(expStart),
                      "malformed exponent");
            return Token(Token::error,
                         llvm::SMLoc::getFromPointer(tokStart),
                         StringRef(tokStart, curPtr - tokStart));
        }

        while (curPtr < bufferEnd && llvm::isDigit(*curPtr))
            ++curPtr;
    }

    return Token(Token::number,
                 llvm::SMLoc::getFromPointer(tokStart),
                 StringRef(tokStart, curPtr - tokStart));
}

Token Lexer::lexStringLiteral() {
    const char *start = curPtr;
    llvm::SMLoc loc = llvm::SMLoc::getFromPointer(start);

    ++curPtr; // eat opening "

    while (curPtr < bufferEnd) {
        char c = *curPtr;

        if (c == '"') {
            ++curPtr; // eat closing "
            llvm::StringRef spelling(start, curPtr - start);
            return Token(Token::string, loc, spelling);
        }

        if (c == '\\') {
            ++curPtr; // eat '\'
            if (curPtr < bufferEnd)
                ++curPtr; // skip escaped char
            continue;
        }

        if (c == '\n' || c == '\r')
            break;

        ++curPtr;
    }

    emitError(loc, "unterminated string literal");
    return Token(Token::error, loc, llvm::StringRef(start, curPtr - start));
}

void Lexer::emitError(llvm::SMLoc loc, llvm::StringRef message) {
    sawError = true;

    /// 尽量走 MLIR 诊断体系：
    /// 1) 通过 SourceMgr 拿到 (line, col)
    /// 2) 构造 FileLineColLoc
    /// 3) emitError 统一打印格式（和后续 parser/IRGen 的错误一致）
    if (context) {
        auto lineCol = sourceMgr.getLineAndColumn(loc, bufferID);
        StringRef fileName =
            sourceMgr.getMemoryBuffer(bufferID)->getBufferIdentifier();

        mlir::Location mlirLoc =
            mlir::FileLineColLoc::get(context,
                                      fileName,
                                      lineCol.first,
                                      lineCol.second);

        mlir::emitError(mlirLoc) << message;
        return;
    }

    /// 没有 MLIRContext 的情况下，退化到 LLVM SourceMgr 的报错输出
    sourceMgr.PrintMessage(loc,
                           llvm::SourceMgr::DK_Error,
                           message);
}

} // namespace ezcompile