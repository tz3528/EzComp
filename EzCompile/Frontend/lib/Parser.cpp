//===-- Parser.cpp ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// 语法分析器实现：将 Token 序列解析为抽象语法树（AST）
//
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringExtras.h"

#include "Parser.h"
#include "AST.h"

namespace ezcompile {

using llvm::StringRef;


//===----------------------------------------------------------------------===//
// Parser - 构造函数和基础方法
//===----------------------------------------------------------------------===//

Parser::Parser(Lexer &lexer, llvm::SourceMgr &sourceMgr, int bufferID, mlir::MLIRContext *ctx)
    : lexer(lexer),
      DiagnosticBase(sourceMgr,bufferID,ctx){
    prevTok = Token(Token::eof, llvm::SMLoc(), llvm::StringRef());
    advance();
}

void Parser::advance() {
    prevTok = curTok;
    curTok = lexer.lex();
}

llvm::SMLoc Parser::tokenEndLoc(const Token &t) {
    const char *p = t.getLoc().getPointer();
    return llvm::SMLoc::getFromPointer(p + t.getSpelling().size());
}

bool Parser::expect(Token::Kind k, StringRef msg) const {
    (void)msg;
    return curTok.is(k);
}

bool Parser::consume(Token::Kind k, StringRef msg) {
    if (!curTok.is(k)) {
        emitError(curTok.getLoc(), msg);
        return false;
    }
    advance();
    return true;
}

void Parser::syncToSectionOrEOF() {
    // 跳过直到遇到下一个 section 关键字或 EOF
    while (!curTok.is(Token::eof)) {
        if (curTok.is(Token::kw_declarations) ||
            curTok.is(Token::kw_equations) ||
            curTok.is(Token::kw_options)) {
            return;
        }
        advance();
    }
}

void Parser::syncToItemEnd() {
    // 跳过直到遇到分号、右花括号或 EOF
    while (!curTok.is(Token::eof) &&
           !curTok.is(Token::semicolon) &&
           !curTok.is(Token::r_brace)) {
        advance();
    }
    if (curTok.is(Token::semicolon)) advance();
}

std::unique_ptr<ModuleAST> Parser::parseModule() {
    llvm::SMLoc begin = curTok.getLoc();
    auto mod = std::make_unique<ModuleAST>(SourceRange(begin, begin));

    while (!curTok.is(Token::eof)) {
        if (curTok.is(Token::kw_declarations)) {
            if (!parseDeclarationsSection(*mod)) syncToSectionOrEOF();
            continue;
        }
        if (curTok.is(Token::kw_equations)) {
            if (!parseEquationsSection(*mod)) syncToSectionOrEOF();
            continue;
        }
        if (curTok.is(Token::kw_options)) {
            if (!parseOptionsSection(*mod)) syncToSectionOrEOF();
            continue;
        }
        emitError(curTok.getLoc(), "expected section keyword: declarations/equations/options");
        syncToSectionOrEOF();
    }

    mod->setRange(SourceRange(begin, curTok.getLoc()));
    return mod;
}

//===----------------------------------------------------------------------===//
// Sections - Section 解析方法
//===----------------------------------------------------------------------===//

bool Parser::parseDeclarationsSection(ModuleAST &mod) {
    if (!consume(Token::kw_declarations, "expected 'declarations'")) return false;
    if (!consume(Token::l_brace, "expected '{' after declarations")) return false;

    while (!curTok.is(Token::r_brace) && !curTok.is(Token::eof)) {
        auto item = parseVarDeclItem();
        if (item) mod.getDecls().push_back(std::move(item));
        else syncToItemEnd();
    }

    return consume(Token::r_brace, "expected '}' to close declarations");
}

bool Parser::parseEquationsSection(ModuleAST &mod) {
    if (!consume(Token::kw_equations, "expected 'equations'")) return false;
    if (!consume(Token::l_brace, "expected '{' after equations")) return false;

    while (!curTok.is(Token::r_brace) && !curTok.is(Token::eof)) {
        auto item = parseEquationItem();
        if (item) mod.getEquations().push_back(std::move(item));
        else syncToItemEnd();
    }

    return consume(Token::r_brace, "expected '}' to close equations");
}

bool Parser::parseOptionsSection(ModuleAST &mod) {
    if (!consume(Token::kw_options, "expected 'options'")) return false;
    if (!consume(Token::l_brace, "expected '{' after options")) return false;

    while (!curTok.is(Token::r_brace) && !curTok.is(Token::eof)) {
        auto item = parseOptionItem();
        if (item) mod.getOptions().push_back(std::move(item));
        else syncToItemEnd();
    }

    return consume(Token::r_brace, "expected '}' to close options");
}

//===----------------------------------------------------------------------===//
// Items - 项解析方法
//===----------------------------------------------------------------------===//

std::unique_ptr<VarDeclAST> Parser::parseVarDeclItem() {
    // 语法格式：var_name[min, max, num]
    if (!curTok.is(Token::identifier)) {
        emitError(curTok.getLoc(), "expected identifier in declaration");
        return nullptr;
    }

    Token nameTok = curTok;
    StringRef name = nameTok.getSpelling();
    llvm::SMLoc begin = nameTok.getLoc();
    advance();

    if (!consume(Token::l_square, "expected '[' after variable name (use var[min,max,num])"))
        return nullptr;

    auto minV = parseSignedNumberLiteral("min");
    if (!minV) return nullptr;

    if (!consume(Token::comma, "expected ',' after min")) return nullptr;

    auto maxV = parseSignedNumberLiteral("max");
    if (!maxV) return nullptr;

    if (!consume(Token::comma, "expected ',' after max")) return nullptr;

    Token numTok = curTok;
    if (!curTok.is(Token::number)) {
        emitError(curTok.getLoc(), "expected integer literal for num");
        return nullptr;
    }
    advance();

    if (!consume(Token::r_square, "expected ']' after num")) return nullptr;

    // 不再支持初始化器
    if (curTok.is(Token::equal)) {
        emitError(curTok.getLoc(), "declaration initializer is not supported; use var[min,max,num] only");
    }

    Token semiTok = curTok;
    if (!consume(Token::semicolon, "expected ';' after declaration")) return nullptr;

    // num 必须是正整数
    long long num = 0;
    if (!llvm::to_integer(numTok.getSpelling(), num) || num <= 0) {
        emitError(numTok.getLoc(), "num must be a positive integer");
        return nullptr;
    }

    return std::make_unique<VarDeclAST>(
        name, std::move(minV), std::move(maxV), num,
        SourceRange(begin, tokenEndLoc(semiTok)));
}

std::unique_ptr<EquationAST> Parser::parseEquationItem() {
    llvm::SMLoc begin = curTok.getLoc();

    auto lhs = parseExpr();
    if (!lhs) return nullptr;

    if (!consume(Token::equal, "expected '=' in equation")) return nullptr;

    auto rhs = parseExpr();
    if (!rhs) return nullptr;

    Token semiTok = curTok;
    if (!consume(Token::semicolon, "expected ';' after equation")) return nullptr;

    return std::make_unique<EquationAST>(
        std::move(lhs), std::move(rhs),
        SourceRange(begin, tokenEndLoc(semiTok)));
}

std::unique_ptr<OptionAST> Parser::parseOptionItem() {
    if (!curTok.is(Token::identifier)) {
        emitError(curTok.getLoc(), "expected option key identifier");
        return nullptr;
    }

    Token keyTok = curTok;
    StringRef key = keyTok.getSpelling();
    llvm::SMLoc begin = keyTok.getLoc();
    advance();

    if (!consume(Token::colon, "expected ':' after option key")) return nullptr;

    auto lit = parseOptionLiteral();
    if (!lit) return nullptr;

    Token semiTok = curTok;
    if (!consume(Token::semicolon, "expected ';' after option")) return nullptr;

    return std::make_unique<OptionAST>(
        key, std::move(lit),
        SourceRange(begin, tokenEndLoc(semiTok)));
}

//===----------------------------------------------------------------------===//
// Expr - 表达式解析（优先级爬升算法）
//===----------------------------------------------------------------------===//

int Parser::getTokPrecedence() const {
    // 运算符优先级：乘除(40) > 加减(20)
    switch (curTok.getKind()) {
    case Token::plus:
    case Token::minus:
        return 20;
    case Token::star:
    case Token::slash:
        return 40;
    default:
        return -1;
    }
}

std::unique_ptr<ExprAST> Parser::parseExpr(int minPrec) {
    auto lhs = parseUnary();
    if (!lhs) return nullptr;
    return parseBinOpRHS(minPrec, std::move(lhs));
}

std::unique_ptr<ExprAST> Parser::parseUnary() {
    if (curTok.is(Token::plus) || curTok.is(Token::minus)) {
        // 防止连续运算符如 ++5, --5, +-5
        if (prevTok.is(Token::plus) || prevTok.is(Token::minus)) {
            emitError(curTok.getLoc(), "consecutive operators are not allowed");
            return nullptr;
        }

        Token opTok = curTok;
        char op = opTok.getSpelling().front();
        llvm::SMLoc begin = opTok.getLoc();
        advance();

        auto operand = parseUnary();
        if (!operand) return nullptr;

        return std::make_unique<UnaryExprAST>(
            op, std::move(operand),
            SourceRange(begin, operand->getEndLoc()));
    }
    return parsePrimary();
}

std::unique_ptr<ExprAST> Parser::parseBinOpRHS(int minPrec,
                                               std::unique_ptr<ExprAST> lhs) {
    // 优先级爬升算法：如果当前运算符优先级低于 minPrec，则返回
    // 否则递归解析右侧表达式
    while (true) {
        int prec = getTokPrecedence();
        if (prec < minPrec) return lhs;

        Token opTok = curTok;
        char op = opTok.getSpelling().empty() ? '?' : opTok.getSpelling().front();
        advance();

        auto rhs = parseUnary();
        if (!rhs) return nullptr;

        int nextPrec = getTokPrecedence();
        // 如果当前运算符优先级小于下一个运算符，先结合 rhs 和后续运算符
        if (prec < nextPrec) {
            rhs = parseBinOpRHS(prec + 1, std::move(rhs));
            if (!rhs) return nullptr;
        }

        auto beginLoc = lhs->getBeginLoc();
        auto endLoc   = rhs->getEndLoc();

        lhs = std::make_unique<BinaryExprAST>(
            op, std::move(lhs), std::move(rhs),
            SourceRange(beginLoc, endLoc));
    }
}

std::unique_ptr<ExprAST> Parser::parsePrimary() {
    // 数字字面量
    if (curTok.is(Token::number)) {
        Token t = curTok;
        advance();

        const StringRef s = t.getSpelling();
        const bool isFloat =
            (s.find('.') != std::string::npos) ||
            (s.find('e') != std::string::npos) ||
            (s.find('E') != std::string::npos);

        if (isFloat) {
            double result;
            s.getAsDouble(result);
            return std::make_unique<FloatExprAST>(result, SourceRange(t.getLoc(), tokenEndLoc(t)));
        } else {
            int64_t result;
            s.getAsInteger(10, result);
            return std::make_unique<IntExprAST>(result,SourceRange(t.getLoc(), tokenEndLoc(t)));
        }
    }

    // 字符串字面量
    if (curTok.is(Token::string)) {
        Token t = curTok;
        advance();

        StringRef sp = t.getSpelling();
        if (sp.size() >= 2 && sp.front() == '"' && sp.back() == '"') {
            sp = sp.drop_front().drop_back();
        }

        return std::make_unique<StringExprAST>(
            sp, SourceRange(t.getLoc(), tokenEndLoc(t)));
    }

    // 标识符（变量引用或函数调用）
    if (curTok.is(Token::identifier)) {
        Token idTok = curTok;
        StringRef name = idTok.getSpelling();
        llvm::SMLoc begin = idTok.getLoc();
        advance();

        if (curTok.is(Token::l_paren)) {
            advance();

            CallExprAST::ArgList args;
            if (!curTok.is(Token::r_paren)) {
                while (true) {
                    auto a = parseExpr();
                    if (!a) return nullptr;
                    args.push_back(std::move(a));
                    if (curTok.is(Token::comma)) {
                        advance();
                        continue;
                    }
                    break;
                }
            } else {
                // 不允许空函数调用
                emitError(curTok.getLoc(), "function calls must have at least one argument");
                return nullptr;
            }

            Token rTok = curTok;
            if (!consume(Token::r_paren, "expected ')'")) return nullptr;

            return std::make_unique<CallExprAST>(
                name, std::move(args),
                SourceRange(begin, tokenEndLoc(rTok)));
        }

        return std::make_unique<VarRefExprAST>(
            name, SourceRange(begin, tokenEndLoc(idTok)));
    }

    // 括号表达式
    if (curTok.is(Token::l_paren)) {
        Token lTok = curTok;
        advance();
        auto inner = parseExpr();
        if (!inner) return nullptr;
        Token rTok = curTok;
        if (!consume(Token::r_paren, "expected ')'")) return nullptr;
        return std::make_unique<ParenExprAST>(
            std::move(inner),
            SourceRange(lTok.getLoc(), tokenEndLoc(rTok)));
    }

    emitError(curTok.getLoc(), "expected expression");
    return nullptr;
}

std::unique_ptr<ExprAST> Parser::parseSignedNumberLiteral(llvm::StringRef what) {
    llvm::SMLoc begin = curTok.getLoc();
    char sign = 0;

    if (curTok.is(Token::plus) || curTok.is(Token::minus)) {
        if (prevTok.is(Token::plus) || prevTok.is(Token::minus)) {
            emitError(curTok.getLoc(), "consecutive operators are not allowed");
            return nullptr;
        }
        sign = curTok.getSpelling().front();
        advance();
    }

    if (!curTok.is(Token::number)) {
        emitError(curTok.getLoc(), ("expected numeric literal for " + what).str());
        return nullptr;
    }

    Token numTok = curTok;
    advance();

    StringRef lit = numTok.getSpelling();
    const bool isFloat =
            (lit.find('.') != std::string::npos) ||
            (lit.find('e') != std::string::npos) ||
            (lit.find('E') != std::string::npos);
    std::unique_ptr<ExprAST> number;
    if (isFloat) {
        double result;
        lit.getAsDouble(result);
        number = std::make_unique<FloatExprAST>(result, SourceRange(begin, tokenEndLoc(numTok)));
    }
    else {
        int64_t result;
        lit.getAsInteger(10, result);
        number = std::make_unique<IntExprAST>(result, SourceRange(begin, tokenEndLoc(numTok)));
    }
    if (sign) {
        return std::make_unique<UnaryExprAST>(
            sign, std::move(number), SourceRange(begin, tokenEndLoc(numTok)));
    }
    return number;
}

//===----------------------------------------------------------------------===//
// Options - 选项字面量解析
//===----------------------------------------------------------------------===//

std::unique_ptr<ExprAST> Parser::parseOptionLiteral() {
    // options 的 value 只允许数字字面量或字符串字面量
    if (curTok.is(Token::plus) || curTok.is(Token::minus) || curTok.is(Token::number)) {
        return parseSignedNumberLiteral("option");
    }

    if (curTok.is(Token::string)) {
        return parsePrimary();
    }

    emitError(curTok.getLoc(), "options only allow numeric/string literals");
    return nullptr;
}


}