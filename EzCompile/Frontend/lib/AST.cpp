//===-- AST.cpp ------------------------------------------------*- C++ -*-===//
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

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include "AST.h"

namespace ezcompile {

static void indent(llvm::raw_ostream &os, unsigned n) {
    for (unsigned i = 0; i < n; ++i)
        os << ' ';
}

static void printExpr(const ExprAST &e, llvm::raw_ostream &os, unsigned ind);
static void printItem(const ItemAST &i, llvm::raw_ostream &os, unsigned ind);

void ExprAST::print(llvm::raw_ostream &os, unsigned indentAmt) const {
    printExpr(*this, os, indentAmt);
}

void ExprAST::dump() const {
    print(llvm::errs(), 0);
}

void ItemAST::print(llvm::raw_ostream &os, unsigned indentAmt) const {
    printItem(*this, os, indentAmt);
}

void ItemAST::dump() const {
    print(llvm::errs(), 0);
}

void ModuleAST::print(llvm::raw_ostream &os, unsigned ind) const {
    indent(os, ind);
    os << "ModuleAST\n";

    indent(os, ind + 4);
    os << "declarations: " << decls.size() << "\n";
    for (const auto &d : decls) {
        if (d)
            d->print(os, ind + 8);
        else {
            indent(os, ind + 8);
            os << "<null decl>\n";
        }
    }

    indent(os, ind + 4);
    os << "equations: " << equations.size() << "\n";
    for (const auto &e : equations) {
        if (e)
            e->print(os, ind + 8);
        else {
            indent(os, ind + 8);
            os << "<null equation>\n";
        }
    }

    indent(os, ind + 4);
    os << "options: " << options.size() << "\n";
    for (const auto &o : options) {
        if (o)
            o->print(os, ind + 8);
        else {
            indent(os, ind + 8);
            os << "<null option>\n";
        }
    }
}

void ModuleAST::dump() const {
    print(llvm::errs(), 0);
}

//===----------------------------------------------------------------------===//
// Internal printers
//===----------------------------------------------------------------------===//

static void printExpr(const ExprAST &e, llvm::raw_ostream &os, unsigned ind) {
    switch (e.getKind()) {
    case ExprAST::Kind::Num: {
        const auto &n = static_cast<const NumExprAST &>(e);
        indent(os, ind);
        os << "NumExprAST literal=" << n.getLiteral() << "\n";
        return;
    }
    case ExprAST::Kind::String: {
        const auto &s = static_cast<const StringExprAST &>(e);
        indent(os, ind);
        os << "StringExprAST value=\"" << s.getValue() << "\"\n";
        return;
    }
    case ExprAST::Kind::VarRef: {
        const auto &v = static_cast<const VarRefExprAST &>(e);
        indent(os, ind);
        os << "VarRefExprAST name=" << v.getName() << "\n";
        return;
    }
    case ExprAST::Kind::UnaryOp: {
        const auto &u = static_cast<const UnaryOpExprAST &>(e);
        indent(os, ind);
        os << "UnaryOpExprAST op='" << u.getOp() << "'\n";
        if (u.getOperand())
            u.getOperand()->print(os, ind + 4);
        else {
            indent(os, ind + 4);
            os << "<null operand>\n";
        }
        return;
    }
    case ExprAST::Kind::BinOp: {
        const auto &b = static_cast<const BinOpExprAST &>(e);
        indent(os, ind);
        os << "BinOpExprAST op='" << b.getOp() << "'\n";
        if (b.getLHS())
            b.getLHS()->print(os, ind + 4);
        else {
            indent(os, ind + 4);
            os << "<null lhs>\n";
        }
        if (b.getRHS())
            b.getRHS()->print(os, ind + 4);
        else {
            indent(os, ind + 4);
            os << "<null rhs>\n";
        }
        return;
    }
    case ExprAST::Kind::Call: {
        const auto &c = static_cast<const CallExprAST &>(e);
        indent(os, ind);
        os << "CallExprAST callee=" << c.getCallee()
           << " args=" << c.getArgs().size() << "\n";
        for (const auto &arg : c.getArgs()) {
            if (arg)
                arg->print(os, ind + 4);
            else {
                indent(os, ind + 4);
                os << "<null arg>\n";
            }
        }
        return;
    }
    }
    llvm_unreachable("unknown ExprAST kind");
}

static void printItem(const ItemAST &i, llvm::raw_ostream &os, unsigned ind) {
    switch (i.getKind()) {
    case ItemAST::Kind::VarDecl: {
        const auto &d = static_cast<const VarDeclAST &>(i);
        indent(os, ind);
        os << "VarDeclAST name=" << d.getName() << "\n";
        if (d.getInit()) {
            indent(os, ind + 4);
            os << "init:\n";
            d.getInit()->print(os, ind + 8);
        } else {
            indent(os, ind + 4);
            os << "init: <default 0>\n";
        }
        return;
    }
    case ItemAST::Kind::Equation: {
        const auto &eq = static_cast<const EquationAST &>(i);
        indent(os, ind);
        os << "EquationAST\n";
        if (eq.getLHS()) {
            indent(os, ind + 4);
            os << "lhs:\n";
            eq.getLHS()->print(os, ind + 8);
        } else {
            indent(os, ind + 4);
            os << "lhs: <null>\n";
        }
        if (eq.getRHS()) {
            indent(os, ind + 4);
            os << "rhs:\n";
            eq.getRHS()->print(os, ind + 8);
        } else {
            indent(os, ind + 4);
            os << "rhs: <null>\n";
        }
        return;
    }
    case ItemAST::Kind::Option: {
        const auto &op = static_cast<const OptionAST &>(i);
        indent(os, ind);
        os << "OptionAST key=" << op.getKey() << "\n";
        if (op.getValue()) {
            indent(os, ind + 4);
            os << "value:\n";
            op.getValue()->print(os, ind + 8);
        } else {
            indent(os, ind + 4);
            os << "value: <null>\n";
        }
        return;
    }
    }
    llvm_unreachable("unknown ItemAST kind");
}

} // namespace ezcompile
