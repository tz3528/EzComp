//===-- AST.h --------------------------------------------------*- C++ -*-===//
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

#ifndef EZ_COMPILE_AST_H
#define EZ_COMPILE_AST_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <memory>

namespace ezcompile {

struct SourceRange {
    llvm::SMLoc begin;
    llvm::SMLoc end;

    SourceRange() = default;
    SourceRange(llvm::SMLoc b, llvm::SMLoc e) : begin(b), end(e) {}
};

class ASTNode {
public:
    SourceRange getRange() const { return range; }
    llvm::SMLoc getBeginLoc() const { return range.begin; }
    llvm::SMLoc getEndLoc() const { return range.end; }

    void setRange(SourceRange r) { range = r; }

protected:
    ASTNode() = default;
    explicit ASTNode(SourceRange r) : range(r) {}

    SourceRange range;
};

//===----------------------------------------------------------------------===//
// ExprAST + LLVM-style RTTI (Kind + classof) + CRTP helpers
//===----------------------------------------------------------------------===//

class ExprAST : public ASTNode {
public:
    enum class Kind : uint8_t {
        Num,
        String,
        VarRef,
        BinOp,
        UnaryOp,
        Call,
    };

    Kind getKind() const { return kind; }

    void print(llvm::raw_ostream &os, unsigned indent = 0) const;
    void dump() const;

protected:
    ExprAST(Kind k, SourceRange r) : ASTNode(r), kind(k) {}

private:
    Kind kind;
};

template <typename Derived, ExprAST::Kind K>
class ExprBase : public ExprAST {
public:
    static bool classof(const ExprAST *e) { return e->getKind() == K; }

protected:
    explicit ExprBase(SourceRange r) : ExprAST(K, r) {}
};

class NumExprAST final
    : public ExprBase<NumExprAST, ExprAST::Kind::Num> {
public:
    NumExprAST(llvm::StringRef literal, SourceRange r)
        : ExprBase(r), literal(literal) {}

    llvm::StringRef getLiteral() const { return literal; }

private:
    llvm::StringRef literal;
};

class StringExprAST final
    : public ExprBase<StringExprAST, ExprAST::Kind::String> {
public:
    StringExprAST(llvm::StringRef value, SourceRange r)
        : ExprBase(r), value(value) {}

    llvm::StringRef getValue() const { return value; }

private:
    llvm::StringRef value;
};

class VarRefExprAST final
    : public ExprBase<VarRefExprAST, ExprAST::Kind::VarRef> {
public:
    VarRefExprAST(llvm::StringRef name, SourceRange r)
        : ExprBase(r), name(name) {}

    llvm::StringRef getName() const { return name; }

private:
    llvm::StringRef name;
};

class UnaryOpExprAST final
    : public ExprBase<UnaryOpExprAST, ExprAST::Kind::UnaryOp> {
public:
    UnaryOpExprAST(char op,
                   std::unique_ptr<ExprAST> operand,
                   SourceRange r)
        : ExprBase(r), op(op), operand(std::move(operand)) {}

    char getOp() const { return op; }
    const ExprAST *getOperand() const { return operand.get(); }
    ExprAST *getOperand() { return operand.get(); }

private:
    char op;
    std::unique_ptr<ExprAST> operand;
};

class BinOpExprAST final
    : public ExprBase<BinOpExprAST, ExprAST::Kind::BinOp> {
public:
    BinOpExprAST(char op,
                 std::unique_ptr<ExprAST> lhs,
                 std::unique_ptr<ExprAST> rhs,
                 SourceRange r)
        : ExprBase(r),
          op(op),
          lhs(std::move(lhs)),
          rhs(std::move(rhs)) {}

    char getOp() const { return op; }
    const ExprAST *getLHS() const { return lhs.get(); }
    const ExprAST *getRHS() const { return rhs.get(); }
    ExprAST *getLHS() { return lhs.get(); }
    ExprAST *getRHS() { return rhs.get(); }

private:
    char op;
    std::unique_ptr<ExprAST> lhs;
    std::unique_ptr<ExprAST> rhs;
};

class CallExprAST final
    : public ExprBase<CallExprAST, ExprAST::Kind::Call> {
public:
    using ArgList = llvm::SmallVector<std::unique_ptr<ExprAST>, 4>;

    CallExprAST(llvm::StringRef callee,
                ArgList args,
                SourceRange r)
        : ExprBase(r),
          callee(callee),
          args(std::move(args)) {}

    llvm::StringRef getCallee() const { return callee; }
    const ArgList &getArgs() const { return args; }
    ArgList &getArgs() { return args; }

private:
    llvm::StringRef callee;
    ArgList args;
};

//===----------------------------------------------------------------------===//
// Top-level items (VarDecl / Equation / Option) + RTTI + CRTP
//===----------------------------------------------------------------------===//

class ItemAST : public ASTNode {
public:
    enum class Kind : uint8_t {
        VarDecl,
        Equation,
        Option
    };

    Kind getKind() const { return kind; }

    void print(llvm::raw_ostream &os, unsigned indent = 0) const;
    void dump() const;

protected:
    ItemAST(Kind k, SourceRange r) : ASTNode(r), kind(k) {}

private:
    Kind kind;
};

template <typename Derived, ItemAST::Kind K>
class ItemBase : public ItemAST {
public:
    static bool classof(const ItemAST *i) { return i->getKind() == K; }

protected:
    explicit ItemBase(SourceRange r) : ItemAST(K, r) {}
};

class VarDeclAST final
    : public ItemBase<VarDeclAST, ItemAST::Kind::VarDecl> {
public:
    VarDeclAST(llvm::StringRef name,
               std::unique_ptr<ExprAST> init,
               SourceRange r)
        : ItemBase(r),
          name(name),
          init(std::move(init)) {}

    llvm::StringRef getName() const { return name; }
    const ExprAST *getInit() const { return init.get(); }
    ExprAST *getInit() { return init.get(); }

private:
    llvm::StringRef name;
    std::unique_ptr<ExprAST> init; // null => default 0
};

class EquationAST final
    : public ItemBase<EquationAST, ItemAST::Kind::Equation> {
public:
    /// LHS 允许是表达式（例如 VarRef 或 diff(...)），语义层再限制“可赋值目标”。
    EquationAST(std::unique_ptr<ExprAST> lhs,
                std::unique_ptr<ExprAST> rhs,
                SourceRange r)
        : ItemBase(r),
          lhs(std::move(lhs)),
          rhs(std::move(rhs)) {}

    const ExprAST *getLHS() const { return lhs.get(); }
    const ExprAST *getRHS() const { return rhs.get(); }
    ExprAST *getLHS() { return lhs.get(); }
    ExprAST *getRHS() { return rhs.get(); }

private:
    std::unique_ptr<ExprAST> lhs;
    std::unique_ptr<ExprAST> rhs;
};

class OptionAST final
    : public ItemBase<OptionAST, ItemAST::Kind::Option> {
public:
    OptionAST(llvm::StringRef key,
              std::unique_ptr<ExprAST> value,
              SourceRange r)
        : ItemBase(r),
          key(key),
          value(std::move(value)) {}

    llvm::StringRef getKey() const { return key; }
    const ExprAST *getValue() const { return value.get(); }
    ExprAST *getValue() { return value.get(); }

private:
    llvm::StringRef key;
    std::unique_ptr<ExprAST> value;
};

//===----------------------------------------------------------------------===//
// ModuleAST (stores 3 vectors directly)
//===----------------------------------------------------------------------===//

class ModuleAST : public ASTNode {
public:
    using DeclList = llvm::SmallVector<std::unique_ptr<VarDeclAST>, 8>;
    using EqList   = llvm::SmallVector<std::unique_ptr<EquationAST>, 8>;
    using OptList  = llvm::SmallVector<std::unique_ptr<OptionAST>, 8>;

    explicit ModuleAST(SourceRange r) : ASTNode(r) {}

    DeclList &getDecls() { return decls; }
    EqList &getEquations() { return equations; }
    OptList &getOptions() { return options; }

    const DeclList &getDecls() const { return decls; }
    const EqList &getEquations() const { return equations; }
    const OptList &getOptions() const { return options; }

    void print(llvm::raw_ostream &os, unsigned indent = 0) const;
    void dump() const;

private:
    DeclList decls;
    EqList equations;
    OptList options;
};

} // namespace ezcompile

#endif // EZ_COMPILE_AST_H
