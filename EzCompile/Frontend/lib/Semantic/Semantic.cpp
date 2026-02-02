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


#include "Semantic/Semantic.h"

#include <queue>

#include "Semantic/DependencyGraph.h"

namespace ezcompile {
std::unique_ptr<SemanticResult> Semantic::analyze(const ModuleAST& module) {
	SymbolTable st;
	OptionsTable opt;
	EquationGroups eg;
	StencilInfo stencil_info;

	auto result = OptionsTable::createWithDefaults();
	if (!result) {
		return nullptr;
	}
	opt = *result;

	collectDecls(module, st);
	if (hadError()) {
		return nullptr;
	}

	checkOptions(module, st, opt);
	if (hadError()) {
		return nullptr;
	}

	checkEquations(module, st, opt, eg);
	if (hadError()) {
		return nullptr;
	}

	TargetFunctionMeta tfm;

	for (size_t i = 0; i < opt.targetFunc.args.size(); i++) {
		auto id = st.lookup(opt.targetFunc.args[i])->id;
		if (i == opt.targetFunc.index) {
			tfm.timeDim = id;
		}
		else {
			tfm.spaceDims.emplace_back(id);
		}
	}
	tfm.func = opt.targetFunc.name;

	adjestEquationOrder(eg, tfm);
	if (hadError()) {
		return nullptr;
	}

	checkStencilInfo(st, eg, tfm, stencil_info);

	return std::make_unique<SemanticResult>(opt, st, eg, tfm, stencil_info);
}

SymbolTable Semantic::collectDecls(const ModuleAST& module, SymbolTable& st) {
	SymbolId id = 0;
	for (const auto& decl : module.getDecls()) {
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

		Domain dom;
		dom.lower = lower;
		dom.upper = upper;
		dom.points = points;
		symbol.domain = dom;

		st.nameToId.emplace(name, id);
		st.symbols.emplace_back(std::move(symbol));
		++id;
	}

	return st;
}

void Semantic::checkEquations(const ModuleAST& module, SymbolTable& st, OptionsTable& opts, EquationGroups& eg) {
	for (auto& equation : module.getEquations()) {
		auto left = equation->getLHS();
		if (auto call = llvm::dyn_cast<CallExprAST>(left)) {
			auto name = call->getCallee().str();

			if (name == opts.targetFunc.name) {
				checkFunctionType(equation.get(), call, st, opts, eg);
			}
			else {
				//非目标函数在等式左侧，为普通方程
				//TODO 这里应该对普通方程也有检查，
				//但是需要加东西，暂时不做处理
				eg.iter.emplace_back(equation.get());
			}
		}
		else {
			//等式左侧不是函数，只能是普通方程
			eg.iter.emplace_back(equation.get());
		}
	}
}

void Semantic::checkFunctionType(
	const EquationAST* equation,
	const CallExprAST* call,
	SymbolTable& st,
	OptionsTable& opts,
	EquationGroups& eg
) {
	std::string err;
	std::string time = *opts.getStr("timeVar", err);

	auto& args = call->getArgs();
	if (call->getArgs().size() != opts.targetFunc.args.size()) {
		//参数不一样多
		emitError(call->getBeginLoc(),
		          "The number of parameters in the targetFunction does not match");
		return;
	}

	//用于记录方程类型
	bool isBoundary = false;
	bool isIteration = false;
	bool isInit = false;
	Anchor anchor;
	for (size_t i = 0; i < opts.targetFunc.args.size(); ++i) {
		auto var = args[i].get();
		auto id = st.lookup(opts.targetFunc.args[i])->id;
		if (i == opts.targetFunc.index) {
			//时间变量的结构必须是t、t+1或字面量常量
			if (auto timeVar = llvm::dyn_cast<VarRefExprAST>(var)) {
				if (timeVar->getName() != time) {
					emitError(var->getBeginLoc(),
					          "The " + std::to_string(i) + "th parameter must be a timeVar");
					return;
				}
			}
			else if (auto timeAddVar = llvm::dyn_cast<BinaryExprAST>(var)) {
				if (timeAddVar->toString() != time + "+1") {
					emitError(var->getBeginLoc(),
					          "The " + std::to_string(i) + "th parameter must be a timeVar");
					return;
				}
				//t+1是迭代方程
				isIteration = true;
			}
			else if (auto num = llvm::dyn_cast<IntExprAST>(var)) {
				auto value = num->getValue();
				auto sb = st.lookup(time);
				if (value < sb->domain.lower || value > sb->domain.upper) {
					emitError(var->getBeginLoc(), "The time variable exceeds the declared range");
					return;
				}
				anchor.dim.emplace_back(id);
				anchor.index.emplace_back(num->getValue());
				isInit = true;
			}
			else {
				//可能是浮点数
				emitError(var->getBeginLoc(),
				          "The " + std::to_string(i) + "th parameter must be a timeVar");
				return;
			}
		}
		else {
			//这里不是时间变量
			if (auto anotherVar = llvm::dyn_cast<VarRefExprAST>(var)) {
				if (anotherVar->getName() != opts.targetFunc.args[i]) {
					emitError(var->getBeginLoc(),
					          "The " + std::to_string(i) + "th parameter does not match");
					return;
				}
			}
			else if (auto num = llvm::dyn_cast<IntExprAST>(var)) {
				auto value = num->getValue();
				auto sb = st.lookup(opts.targetFunc.args[i]);
				if (value < sb->domain.lower || value > sb->domain.upper) {
					emitError(var->getBeginLoc(), "The time variable exceeds the declared range");
					return;
				}
				anchor.dim.emplace_back(id);
				anchor.index.emplace_back(num->getValue());
				isBoundary = true;
			}
			else if (auto bop = llvm::dyn_cast<BinaryExprAST>(var)) {
				auto op = bop->getOp();
				auto lhs = bop->getLHS();
				auto rhs = bop->getRHS();
				if (op != '+' && op != '-') {
					emitError(var->getBeginLoc(), "The operator must is '+' or '-'");
					return;
				}

				if (auto selfVar = llvm::dyn_cast<VarRefExprAST>(lhs)) {
					if (selfVar->getName() != opts.targetFunc.args[i]) {
						emitError(lhs->getBeginLoc(),
						          "The " + std::to_string(i) + "th parameter does not match");
						return;
					}
					if (!llvm::isa<IntExprAST>(rhs)) {
						emitError(rhs->getBeginLoc(), "rhs must is a Interger");
						return;
					}
					isIteration = true;
				}
				else {
					emitError(lhs->getBeginLoc(), "The left must is a var");
					return;
				}
			}
			else {
				emitError(call->getBeginLoc(), "function is undefine");
			}
		}
	}

	if (isIteration && !isBoundary && !isInit) {
		eg.iter.emplace_back(equation);
	}
	else if (!isIteration && isBoundary && !isInit) {
		if (anchor.dim.empty()) {
			emitError(equation->getBeginLoc(), "anchor is uninitialized");
		}
		eg.boundary.emplace_back(EquationAnchor{equation, anchor});
	}
	else if (!isIteration && !isBoundary && isInit) {
		if (anchor.dim.empty()) {
			emitError(equation->getBeginLoc(), "anchor is uninitialized");
		}
		eg.init.emplace_back(EquationAnchor{equation, anchor});
	}
	else {
		emitError(equation->getBeginLoc(), "equation have more than two type");
	}
}


void Semantic::checkOptions(const ModuleAST& module, SymbolTable& st, OptionsTable& opts) {
	std::string err;
	mlir::LogicalResult result = mlir::success();
	int64_t valInt;
	double valDouble;

	for (auto& option : module.getOptions()) {
		auto key = option->getKey().str();
		auto expr = option->getValue();

		if (key == "function") {
			checkFunction(expr, st, opts);
			continue;
		}

		if (auto value = llvm::dyn_cast<StringExprAST>(expr)) {
			result = opts.set(key, value->getValue().str(), err);
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

	auto timeVar = *opts.getStr("timeVar", err);

	if (st.lookup(timeVar) == nullptr) {
		emitError(module.getEndLoc(), "timeVar undeclared");
		return;
	}

	for (size_t i = 0; i < opts.targetFunc.args.size(); ++i) {
		if (opts.targetFunc.args[i] == timeVar) {
			if (opts.targetFunc.index == -1) {
				opts.targetFunc.index = i;
			}
			else {
				emitError(module.getEndLoc(), "targetFunction have two timeVar");
				return;
			}
		}
	}
	if (opts.targetFunc.index == -1) {
		emitError(module.getEndLoc(), "targetFunction no timeVar");
	}
}

void Semantic::checkFunction(ExprAST* expr, SymbolTable& st, OptionsTable& opts) {
	auto e = llvm::dyn_cast<StringExprAST>(expr);
	if (!e) {
		emitError(expr->getBeginLoc(), "Function name must be a string");
		return;
	}

	std::string function = e->getValue().str();
	opts.targetFunc.text = function;

	auto left = function.find('(');
	auto right = function.find(')');

	if (left == 0) {
		emitError(expr->getBeginLoc(), "No function name");
		return;
	}
	if (right > function.size() - 1) {
		emitError(expr->getBeginLoc(), "There are extra characters");
		return;
	}

	std::string name = function.substr(0, left);
	opts.targetFunc.name = name;

	size_t begin = left + 1;
	size_t end = left + 1;
	size_t count = 0;

	while (end <= right) {
		while (function[end] != ',' && end < right) end++;
		auto arg = function.substr(begin, end - begin);

		if (st.lookup(arg) == nullptr) {
			emitError(expr->getBeginLoc(), arg + " is an undeclared variable in the function");
			return;
		}

		opts.targetFunc.args.emplace_back(arg);
		count++;
		begin = end + 1;
		end = begin;
	}
}

void Semantic::checkStencilInfo(SymbolTable& st, EquationGroups& eg, TargetFunctionMeta& target,
                                StencilInfo & stencil_info) {
	for (auto& eq : eg.iter) {
		auto lhs = eq->getLHS();
		auto rhs = eq->getRHS();

		walkExpr(lhs, [&](const ExprAST* lhs) {
			checkShiftInfo(lhs, st, target, stencil_info);
			return true;
		});
		walkExpr(rhs, [&](const ExprAST* rhs) {
			checkShiftInfo(rhs, st, target, stencil_info);
			return true;
		});
	}
}

void Semantic::checkShiftInfo(const ExprAST* expr, SymbolTable& st, TargetFunctionMeta& target,
                              StencilInfo & stencil_info) {
	if (auto call = llvm::dyn_cast<CallExprAST>(expr)) {
		auto func = call->getCallee().str();
		if (func != target.func) return;

		auto& args = call->getArgs();
		ShiftInfo info;
		for (size_t i = 0; i < args.size(); i++) {
			// 这里除了时间变量外，其余变量均要枚举
			if (auto bop = llvm::dyn_cast<BinaryExprAST>(args[i].get())) {
				auto op = bop->getOp();
				auto lhs = bop->getLHS();
				auto rhs = bop->getRHS();
				if (auto var = llvm::dyn_cast<VarRefExprAST>(lhs)) {
					if (auto num = llvm::dyn_cast<IntExprAST>(rhs)) {
						auto value = num->getValue();
						auto name = var->getName().str();
						auto id = st.lookup(name)->id;

						// 变量 +或- 常量，说明符合偏移模式
						if (op == '+') {
							info.dim.emplace_back(id);
							info.offset.emplace_back(value);
						}
						else if (op == '-') {
							info.dim.emplace_back(id);
							info.offset.emplace_back(-value);
						}
					}
				}
			}
			else if (auto var = llvm::dyn_cast<VarRefExprAST>(args[i].get())) {
				auto name = var->getName().str();
				auto id = st.lookup(name)->id;
				info.dim.emplace_back(id);
				info.offset.emplace_back(0);
			}
		}
		stencil_info.call_info.emplace(call, info);
		for (size_t i = 0; i < info.dim.size(); ++i) {
			stencil_info.symbol_info[info.dim[i]].insert(info.offset[i]);
		}
		stencil_info.shift_infos.insert(info);
	}
}

void Semantic::adjestEquationOrder(EquationGroups& eg, TargetFunctionMeta& target) {
	EqGraph G;

	// 首先收集所有方程的左侧定义
	llvm::StringMap<const EquationAST*> definite;
	for (auto eq : eg.iter) {
		EqNode node;
		node.eq = eq;
		G.addNode(node);

		auto lhs = eq->getLHS();

		if (definite.find(lhs->getSourceText()) != definite.end()) {
			emitError(eq->getBeginLoc(), "The expression on the left side of the equation cannot be repeated");
		}

		if (auto call = llvm::dyn_cast<CallExprAST>(lhs)) {
			if (call->getCallee().str() != target.func) {
				definite[call->getSourceText()] = eq;
			}
		}
	}

	// 枚举所有方程，找到每个方程对应的依赖方程
	for (auto& eq : eg.iter) {
		walkExpr(eq->getRHS(), [&](const ExprAST* expr) {
			auto name = expr->getSourceText();
			if (definite.contains(name)) {
				G.addEdge(definite[name], eq);
			}
			return true;
		});
	}

	// 用当前依赖图的bfs更新远方程序
	auto result = G.getTopoOrder();
	if (mlir::failed(result)) {
		emitError(eg.iter[0]->getBeginLoc(), "There must be no circular dependencies.");
	}

	for (size_t i = 0; i < result->size(); i++) {
		eg.iter[i] = result->at(i);
	}

	outputDotGraph(G, "Dependency Graph");
}

bool Semantic::getInteger(ExprAST* expr, int64_t& result) {
	if (auto* num = llvm::dyn_cast<IntExprAST>(expr)) {
		result = num->getValue();
		return true;
	}
	if (auto* unNum = llvm::dyn_cast<UnaryExprAST>(expr)) {
		if (auto* num = llvm::dyn_cast<IntExprAST>(unNum->getOperand())) {
			result = num->getValue();
			if (unNum->getOp() == '-') {
				result = -result;
			}
			return true;
		}
	}
	return false;
}

bool Semantic::getFloat(ExprAST* expr, double& result) {
	if (auto* num = llvm::dyn_cast<FloatExprAST>(expr)) {
		result = num->getValue();
		return true;
	}
	if (auto* unNum = llvm::dyn_cast<UnaryExprAST>(expr)) {
		if (auto* num = llvm::dyn_cast<FloatExprAST>(unNum->getOperand())) {
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
