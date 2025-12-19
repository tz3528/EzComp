//===-- Parser.cpp ---------------------------------------------*- C++ -*-===//
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

#include "Parser.h"
#include "AST.h"

namespace ezcompile {

using llvm::StringRef;

//===----------------------------------------------------------------------===//
// OptionRegistry
//===----------------------------------------------------------------------===//

void OptionRegistry::addNumber(StringRef key, llvm::ArrayRef<StringRef> allowed) {
    Rule r;
    r.kind = ValueKind::Number;
    for (auto v : allowed) r.allowed.push_back(v.str());
    rules[key] = std::move(r);
}

void OptionRegistry::addString(StringRef key, llvm::ArrayRef<StringRef> allowed) {
    Rule r;
    r.kind = ValueKind::String;
    for (auto v : allowed) r.allowed.push_back(v.str());
    rules[key] = std::move(r);
}

const OptionRegistry::Rule *OptionRegistry::lookup(StringRef key) const {
    auto it = rules.find(key);
    if (it == rules.end()) return nullptr;
    return &it->second;
}

//===----------------------------------------------------------------------===//
// Parser
//===----------------------------------------------------------------------===//

Parser::Parser(Lexer &lexer, llvm::SourceMgr &sourceMgr, int bufferID, mlir::MLIRContext *ctx)
    : lexer(lexer),
      sourceMgr(sourceMgr),
      bufferID(bufferID),
      ctx(ctx),
      saver(alloc) {
    advance();

    // 默认 options 规则（可在外部增删改）
    options.setAllowUnknown(false);
    options.addNumber("precision");
    options.addNumber("delta");
    options.addNumber("length");
}

void Parser::advance() {
    curTok = lexer.lex();
}

llvm::SMLoc Parser::tokenEndLoc(const Token &t) {
    const char *p = t.getLoc().getPointer();
    return llvm::SMLoc::getFromPointer(p + t.getSpelling().size());
}

void Parser::emitError(llvm::SMLoc loc, StringRef msg) {
    sawError = true;
    if (!ctx) {
        sourceMgr.PrintMessage(loc, llvm::SourceMgr::DK_Error, msg);
        return;
    }
    auto lineCol = sourceMgr.getLineAndColumn(loc, bufferID);
    auto *buf = sourceMgr.getMemoryBuffer(bufferID);
    StringRef file = buf ? buf->getBufferIdentifier() : "<input>";
    auto flc = mlir::FileLineColLoc::get(ctx, file, lineCol.first, lineCol.second);
    mlir::emitError(flc) << msg;
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
// Sections
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
// Items
//===----------------------------------------------------------------------===//

std::unique_ptr<VarDeclAST> Parser::parseVarDeclItem() {
    if (!curTok.is(Token::identifier)) {
        emitError(curTok.getLoc(), "expected identifier in declaration");
        return nullptr;
    }

    Token nameTok = curTok;
    StringRef name = nameTok.getSpelling();
    llvm::SMLoc begin = nameTok.getLoc();
    advance();

    std::unique_ptr<ExprAST> init;
    if (curTok.is(Token::equal)) {
        advance();
        init = parseExpr();
        if (!init) return nullptr;
    }

    Token semiTok = curTok;
    if (!consume(Token::semicolon, "expected ';' after declaration")) return nullptr;

    return std::make_unique<VarDeclAST>(
        name, std::move(init),
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

    validateOption(keyTok, lit.get());

    Token semiTok = curTok;
    if (!consume(Token::semicolon, "expected ';' after option")) return nullptr;

    return std::make_unique<OptionAST>(
        key, std::move(lit),
        SourceRange(begin, tokenEndLoc(semiTok)));
}

//===----------------------------------------------------------------------===//
// Expr (precedence climbing)
//===----------------------------------------------------------------------===//

int Parser::getTokPrecedence() const {
    switch (curTok.getKind()) {
    case Token::plus:
    case Token::minus:
        return 10;
    case Token::star:
    case Token::slash:
        return 20;
    default:
        return -1;
    }
}

std::unique_ptr<ExprAST> Parser::parseExpr(int minPrec) {
    auto lhs = parseUnary();
    if (!lhs) return nullptr;

    while (true) {
        int prec = getTokPrecedence();
        if (prec < minPrec) break;

        Token opTok = curTok;
        char op = opTok.getSpelling().empty() ? '?' : opTok.getSpelling().front();
        advance();

        auto rhs = parseUnary();
        if (!rhs) return nullptr;

        int nextPrec = getTokPrecedence();
        if (prec < nextPrec) {
            rhs = parseExpr(prec + 1);
            if (!rhs) return nullptr;
        }

        lhs = std::make_unique<BinaryExprAST>(
            op, std::move(lhs), std::move(rhs),
            SourceRange(lhs->getBeginLoc(), rhs->getEndLoc()));
    }

    return lhs;
}

std::unique_ptr<ExprAST> Parser::parseUnary() {
    if (curTok.is(Token::plus) || curTok.is(Token::minus)) {
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

std::unique_ptr<ExprAST> Parser::parsePrimary() {
    if (curTok.is(Token::number)) {
        Token t = curTok;
        advance();
        return std::make_unique<NumberExprAST>(
            t.getSpelling(),
            SourceRange(t.getLoc(), tokenEndLoc(t)));
    }

    if (curTok.is(Token::string)) {
        Token t = curTok;
        advance();

        StringRef sp = t.getSpelling();
        if (sp.size() >= 2 && sp.front() == '"' && sp.back() == '"') {
            sp = sp.drop_front().drop_back();
        }
        sp = intern(sp);

        return std::make_unique<StringExprAST>(
            sp, SourceRange(t.getLoc(), tokenEndLoc(t)));
    }

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

    if (curTok.is(Token::l_paren)) {
        Token lTok = curTok;
        advance();
        auto inner = parseExpr();
        if (!inner) return nullptr;
        Token rTok = curTok;
        if (!consume(Token::r_paren, "expected ')'")) return nullptr;
        inner->setRange(SourceRange(lTok.getLoc(), tokenEndLoc(rTok)));
        return inner;
    }

    emitError(curTok.getLoc(), "expected expression");
    return nullptr;
}

//===----------------------------------------------------------------------===//
// options: literal-only + validation
//===----------------------------------------------------------------------===//

std::unique_ptr<ExprAST> Parser::parseOptionLiteral() {
    // number literal (允许前导 +/-，但整体仍视为“数字字面量”)
    if (curTok.is(Token::plus) || curTok.is(Token::minus) || curTok.is(Token::number)) {
        llvm::SMLoc begin = curTok.getLoc();
        char sign = 0;

        if (curTok.is(Token::plus) || curTok.is(Token::minus)) {
            sign = curTok.getSpelling().front();
            advance();
        }

        if (!curTok.is(Token::number)) {
            emitError(curTok.getLoc(), "expected numeric literal");
            return nullptr;
        }

        Token numTok = curTok;
        advance();

        StringRef lit = numTok.getSpelling();
        auto number = std::make_unique<NumberExprAST>(lit, SourceRange(begin, tokenEndLoc(numTok)));
        if (sign) {
            return std::make_unique<UnaryExprAST>(
                sign, std::move(number),SourceRange(begin, tokenEndLoc(numTok)));
        }

        return number;
    }

    // string literal
    if (curTok.is(Token::string)) {
        return parsePrimary(); // primary 已经做了 strip + intern
    }

    emitError(curTok.getLoc(), "options only allow numeric/string literals");
    return nullptr;
}

void Parser::validateOption(const Token &keyTok, const ExprAST *value) {
    StringRef key = keyTok.getSpelling();
    const auto *rule = options.lookup(key);

    if (!rule) {
        if (!options.allowUnknownKeys()) {
            emitError(keyTok.getLoc(), "unknown option key: '" + key.str() + "'");
        }
        return;
    }

    if (rule->kind == OptionRegistry::ValueKind::Number && !isNumberLiteral(value)) {
        emitError(keyTok.getLoc(), "option '" + key.str() + "' expects number literal");
        return;
    }
    if (rule->kind == OptionRegistry::ValueKind::String && !isStringLiteral(value)) {
        emitError(keyTok.getLoc(), "option '" + key.str() + "' expects string literal");
        return;
    }

    if (rule->allowed.empty()) return;

    std::string got;
    if (auto *n = llvm::dyn_cast<NumberExprAST>(value)) got = n->getLiteral().str();
    else if (auto *s = llvm::dyn_cast<StringExprAST>(value)) got = s->getValue().str();

    for (auto &ok : rule->allowed) {
        if (got == ok) return;
    }

    emitError(keyTok.getLoc(), "invalid value for option '" + key.str() + "'");
}

}
