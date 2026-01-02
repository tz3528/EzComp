//===-- Semantic.cpp -------------------------------------------*- C++ -*-===//
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

#include "Semantic.h"

namespace ezcompile {

std::unique_ptr<SemanticResult> Semantic::analyze(const ModuleAST& module) {
	SymbolTable st;
	OptionsTable opt;
	EquationGroups eg;
	collectDecls(module, st);
	checkOptions(module, st, opt);
	checkEquations(module);
}

SymbolTable Semantic::collectDecls(const ModuleAST& module, SymbolTable& st) {
	SymbolId id = 0;
	for (const auto &decl : module.getDecls()) {
		auto name = decl->getName().str();

		if (st.nameToId.find(name) != st.nameToId.end()) {
			emitError(decl->getBeginLoc(), "Duplicate declaration: " + name);
			continue;
		}

		auto minv = decl->getMinV();
		auto maxv = decl->getMaxV();
		auto points = decl->getNum();

		int64_t lower, upper;
		if (!getInteger(minv, lower)) {
			emitError(minv->getBeginLoc(), "The lower bound is not an integer");
			continue;
		}

		if (!getInteger(maxv, upper)) {
			emitError(maxv->getBeginLoc(), "The upper bound is not an integer");
			continue;
		}

		if (lower > upper) {
			emitError(minv->getBeginLoc(), "The lower bound is larger than the upper bound");
			continue;
		}

		if (points <= 1) {
			emitError(minv->getBeginLoc(), "The number of sampling points should be at least 2");
			continue;
		}

		Symbol symbol;
		symbol.id = id;
		symbol.name = name;
		symbol.kind = SymbolKind::IndependentVar;

		Domain dom;
		dom.lower = lower;
		dom.upper = upper;
		dom.points = points;
		dom.step = static_cast<double>(upper - lower) / static_cast<double>(points - 1);
		symbol.domain = dom;

		st.nameToId.emplace(name, id);
		st.symbols.emplace_back(std::move(symbol));
		++id;
	}

	return st;
}

void Semantic::checkEquations(const ModuleAST& module) {

}

void Semantic::checkOptions(const ModuleAST& module, SymbolTable& st, OptionsTable& opts) {
	std::string err;
	mlir::LogicalResult result = mlir::success();
	int64_t valInt;
	double valDouble;

	for (auto & option : module.getOptions()) {
		auto key = option->getKey().str();
		auto expr = option->getValue();

		if (auto value = llvm::dyn_cast<StringExprAST>(expr)) {
			result = opts.set(key, value->getValue().str(),err);
		}
		else if (getInteger(expr, valInt)) {
			result = opts.set(key, valInt, err);
		}
		else if (getFloat(expr, valDouble)) {
			result = opts.set(key, valDouble, err);
		}

		if (failed(result)) {
			emitError(expr->getBeginLoc(), err);
		}
	}
}

bool Semantic::getInteger(ExprAST* expr, int64_t &result) {
	if (auto *num = llvm::dyn_cast<IntExprAST>(expr)) {
		result = num->getValue();
		return true;
	}
	if (auto *unNum = llvm::dyn_cast<UnaryExprAST>(expr)) {
		if (auto *num = llvm::dyn_cast<IntExprAST>(unNum->getOperand())) {
			result = num->getValue();
			if (unNum->getOp() == '-') {
				result = -result;
			}
			return true;
		}
	}
	return false;
}

bool Semantic::getFloat(ExprAST* expr, double &result) {
	if (auto *num = llvm::dyn_cast<FloatExprAST>(expr)) {
		result = num->getValue();
		return true;
	}
	if (auto *unNum = llvm::dyn_cast<UnaryExprAST>(expr)) {
		if (auto *num = llvm::dyn_cast<FloatExprAST>(unNum->getOperand())) {
			result = num->getValue();
			if (unNum->getOp() == '-') {
				result = -result;
			}
			return true;
		}
	}
	return false;
}


}
