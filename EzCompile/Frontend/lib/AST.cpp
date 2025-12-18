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

#include <complex>

namespace ezcompile {

void ASTDumper::dump(ExprAST *expr) {
    llvm::TypeSwitch<ExprAST *>(expr)
          .Case<BinaryExprAST, CallExprAST, StringExprAST,
                NumberExprAST, UnaryExprAST, VarRefExprAST>(
              [&](auto *node) { this->dump(node); })
          .Default([&](ExprAST *) {
            // No match, fallback to a generic message
            INDENT();
            llvm::errs() << "<unknown Expr, kind " << expr->getKind() << ">\n";
          });
}

void ASTDumper::dump(VarDeclAST *node) {
    INDENT();
    llvm::errs() << "VarDecl : " << node->getName();
    if (node->getInit() != nullptr) {
        llvm::errs() << " = ";
        dump(node->getInit());
    }
    llvm::errs() << "\n";
}

void ASTDumper::dump(EquationAST *node) {
    INDENT();
    llvm::errs() << "Equation : ";
    dump(node->getLHS());
    llvm::errs() << " = ";
    dump(node->getRHS());
    llvm::errs() << "\n";
}

void ASTDumper::dump(OptionAST *node) {
    INDENT();
    llvm::errs() << "Option :\n";
    llvm::errs() << "    Key{ " << node->getKey() << " }\n";
    llvm::errs() << "    Value{ " << node->getValue() << " }\n";
}

void ASTDumper::dump(NumberExprAST *node) {
    INDENT();
    llvm::errs() << node->getLiteral();
}

void ASTDumper::dump(StringExprAST *node) {
    INDENT();
    llvm::errs() << node->getValue();
}

void ASTDumper::dump(VarRefExprAST *node) {
    INDENT();
    llvm::errs() << node->getName();
}

void ASTDumper::dump(UnaryExprAST *node) {
    INDENT();
    llvm::errs() << node->getOp() << node->getOperand();
}

void ASTDumper::dump(BinaryExprAST *node) {
    INDENT();
    llvm::errs() << node->getLHS() << " = " << node->getRHS();
}

void ASTDumper::dump(CallExprAST *node) {
    INDENT();
    llvm::errs() << node->getCallee();
    llvm::errs() << "(";
    for (auto it = node->getArgs().begin(); it != node->getArgs().end(); ) {
        dump((*it).get());
        it++;
        if (it != node->getArgs().end()) {
            llvm::errs() << ", ";
        }
    }
    llvm::errs() << ")\n";
}

void ASTDumper::dump(ModuleAST *module) {
    llvm::errs() << "Declarations:\n";
    for (auto& decl : module->getDecls()) {
        dump(decl.get());
    }
    llvm::errs() << "\nEquations:\n";
    for (auto& equation : module->getEquations()) {
        dump(equation.get());
    }
    llvm::errs() << "\nOptions:\n";
    for (auto& option : module->getOptions()) {
        dump(option.get());
    }
    llvm::errs() << "\n";
}

void dump(ModuleAST &module ){ ASTDumper().dump(&module); }

} // namespace ezcompile
