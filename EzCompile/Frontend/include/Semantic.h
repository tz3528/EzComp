//===-- Semantic.h ---------------------------------------------*- C++ -*-===//
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


#ifndef EZ_COMPILE_SEMANTIC_H
#define EZ_COMPILE_SEMANTIC_H

#include "Parser.h"
#include "DiagnosticBase.h"
#include "OptionTable.h"

namespace ezcompile {
// ====== SemanticResult ===============

using SymbolId = uint64_t;

// ------------------------------
// 2) 符号表 SymbolTable
// ------------------------------
// declarations: IDENT "[" INT "," INT "," INT "]" ";"
// 三个参数分别：下界、上界、区间点数
struct Domain {
	int64_t lower = 0.0; // 下界
	int64_t upper = 0.0; // 上界
	uint64_t points = 0;  // 点数 N
};

struct Symbol {
	SymbolId id = 0;
	std::string name;
	Domain domain;
};

struct SymbolTable {
	std::unordered_map<std::string, SymbolId> nameToId;
	std::vector<Symbol> symbols;

	const Symbol* lookup(const std::string& name) const {
		auto it = nameToId.find(name);
		if (it == nameToId.end()) return nullptr;
		return &symbols[it->second];
	}

	const Symbol& get(SymbolId id) const { return symbols[id]; }
};

struct Anchor {
	SymbolId dim = 0;                 // 被固定的维度
	uint64_t index;     // 若能落到均匀网格索引：0 或 N-1（或 t 的 0）
};

struct EquationAnchor {
	const EquationAST* eq;
	Anchor anchor;
};

// ------------------------------------------------------
// 3/4/5) 方程分组 EquationGroups
// ------------------------------------------------------
struct EquationGroups {
	// 初始化方程列表：如 u(x,0)=0;
	std::vector<EquationAnchor> init;

	// 边界方程列表：如 u(0,t)=10; u(100,t)=10;
	std::vector<EquationAnchor> boundary;

	// 每次迭代的普通方程：如 PDE 主方程/离散模板等
	std::vector<const EquationAST*> iter;
};

// ------------------------------------------------------
// TargetFunctionMeta
// - 目标未知函数符号（u）
// - 时间维符号（t）
// - 空间维符号（x、y…）
struct TargetFunctionMeta {
	SymbolId funcSym = 0;            // u
	SymbolId timeDim = 0;            // t
	std::vector<SymbolId> spaceDims; // [x] 或 [x,y]
};

// ------------------------------------------------------
// SemanticResult
// ------------------------------------------------------
struct SemanticResult {
	// 1) 选项表（Core）
	OptionsTable options;

	// 2) 符号表（Core）
	SymbolTable symbols;

	// 3/4/5) 初始化/边界/迭代方程指针列表（Core）
	EquationGroups eqs;

	// Optional：目标函数与维度角色
	std::optional<TargetFunctionMeta> target;
};

/// 用于存储解析结果和语法分析中的内容
struct ParsedModule {
	llvm::SourceMgr sourceMgr;
	int bufferID = 0;
	std::unique_ptr<ModuleAST> module;
	std::unique_ptr<SemanticResult> sema;
};


class Semantic : public DiagnosticBase {
public:
	Semantic(llvm::SourceMgr& sourceMgr, int bufferID, mlir::MLIRContext* ctx)
		: DiagnosticBase(sourceMgr, bufferID, ctx) {
	};

	std::unique_ptr<SemanticResult> analyze(const ModuleAST& module);

private:
	SymbolTable collectDecls(const ModuleAST& module, SymbolTable& st);
	void checkOptions(const ModuleAST& module, SymbolTable& st, OptionsTable& opts);
	void checkEquations(const ModuleAST& module, SymbolTable& st, OptionsTable& opts, EquationGroups &eg);
	void checkFunctionType(const EquationAST * equation,const CallExprAST * call, SymbolTable& st, OptionsTable& opts, EquationGroups &eg);
	void checkFunction(ExprAST* expr, SymbolTable& st, OptionsTable& opts);

	static bool getInteger(ExprAST* expr, int64_t &result);
	static bool getFloat(ExprAST* expr, double &result);
};

}

#endif //EZ_COMPILE_SEMANTIC_H
