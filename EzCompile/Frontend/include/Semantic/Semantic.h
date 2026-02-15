//===-- Semantic.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// 语义分析接口
// 该文件定义了语义分析阶段的数据结构和接口，包括：
// - 符号表管理
// - 方程分类和验证
// - 模板（Stencil）信息提取
// - 方程依赖关系分析
//
//===----------------------------------------------------------------------===//


#ifndef EZ_COMPILE_SEMANTIC_H
#define EZ_COMPILE_SEMANTIC_H

#include "Parser.h"
#include "DiagnosticBase.h"
#include "OptionTable.h"

namespace ezcompile {

using SymbolId = uint64_t;

// ------------------------------
// 符号表 SymbolTable
// ------------------------------
// declarations: IDENT "[" INT "," INT "," INT "]" ";"
// 三个参数分别：下界、上界、区间点数

/// 变量的定义域，表示均匀离散网格的范围
struct Domain {
	int64_t lower = 0.0; // 下界
	int64_t upper = 0.0; // 上界
	uint64_t points = 0; // 网格点数
};

/// 符号表条目，表示一个声明的变量
struct Symbol {
	SymbolId id = 0;
	std::string name;
	Domain domain;
};

/// 符号表，管理所有变量声明
struct SymbolTable {
	std::unordered_map<std::string, SymbolId> nameToId;
	std::vector<Symbol> symbols;

	/// 根据名称查找符号，不存在返回nullptr
	const Symbol* lookup(const std::string& name) const {
		auto it = nameToId.find(name);
		if (it == nameToId.end()) return nullptr;
		return &symbols[it->second];
	}

	const Symbol& get(SymbolId id) const { return symbols[id]; }
};

/// 锚点信息，记录方程中被固定的维度
struct Anchor {
	std::vector<SymbolId> dim;    // 被固定的维度ID
	std::vector<uint64_t> index;  // 对应的固定索引值
};

/// 带锚点的方程，用于初始化和边界条件
struct EquationAnchor {
	const EquationAST* eq;
	Anchor anchor;
};

/**
 * 偏移信息，记录函数调用中各个维度的偏移量
 * 例如：u(x+1,t) 中，x维度偏移+1，t维度偏移0
 */
struct ShiftInfo {
	llvm::SmallVector<SymbolId, 4> dim;
	llvm::SmallVector<int64_t, 4> offset;

	bool operator==(const ShiftInfo& other) const { return dim == other.dim && offset == other.offset; }
	bool operator<(const ShiftInfo& other) const { return dim != other.dim ? dim < other.dim : offset < other.offset; }
};

/// 模板（Stencil）信息，记录函数调用中的偏移模式
struct StencilInfo {
	// 每个变量在所有调用中出现的偏移量集合
	std::map<SymbolId, std::set<int64_t>> symbol_info;

	// 每个函数调用对应的完整偏移信息
	std::map<const CallExprAST*, ShiftInfo> call_info;

	// 所有偏移模式的集合
	std::set<ShiftInfo> shift_infos;
};

// ------------------------------------------------------
// 方程分组 EquationGroups
// ------------------------------------------------------

/// 方程分组，根据类型分为初始化、边界和迭代三类
struct EquationGroups {
	// 初始化方程：如 u(x,0)=0
	std::vector<EquationAnchor> init;

	// 边界方程：如 u(0,t)=10
	std::vector<EquationAnchor> boundary;

	// 迭代方程：如 PDE 主方程
	std::vector<const EquationAST*> iter;
};

// ------------------------------------------------------
// 目标函数元数据
// ------------------------------------------------------

/// 目标函数元数据，记录函数名和维度角色
struct TargetFunctionMeta {
	std::string func;                     // 函数名
	SymbolId timeDim = 0;                 // 时间维度
	std::vector<SymbolId> spaceDims;      // 空间维度列表
};

// ------------------------------------------------------
// 语义分析结果
// ------------------------------------------------------
struct SemanticResult {
	OptionsTable options;           // 选项表
	SymbolTable st;                 // 符号表
	EquationGroups egs;             // 方程分组
	TargetFunctionMeta target;      // 目标函数元数据
	StencilInfo stencil_info;       // 模板信息
};

/// 解析模块，包含AST和语义分析结果
struct ParsedModule {
	llvm::SourceMgr sourceMgr;
	int bufferID = 0;
	std::unique_ptr<ModuleAST> module;
	std::unique_ptr<SemanticResult> sema;
};


class Semantic : public DiagnosticBase {
public:
	Semantic(llvm::SourceMgr& sourceMgr, int bufferID, mlir::MLIRContext* ctx)
		: DiagnosticBase(sourceMgr, bufferID, ctx) {};

	/// 主入口：对模块进行语义分析
	std::unique_ptr<SemanticResult> analyze(const ModuleAST& module);

private:
	/// 收集变量声明，构建符号表
	SymbolTable collectDecls(const ModuleAST& module, SymbolTable& st);

	/// 检查选项配置
	void checkOptions(const ModuleAST& module, SymbolTable& st, OptionsTable& opts);

	/// 检查方程并进行分类
	void checkEquations(const ModuleAST& module, SymbolTable& st, OptionsTable& opts, EquationGroups& eg);

	/// 检查函数调用类型（初始化/边界/迭代）
	void checkFunctionType(const EquationAST* equation, const CallExprAST* call, SymbolTable& st, 
	                      OptionsTable& opts, EquationGroups& eg);

	/// 解析目标函数字符串的函数信息
	void checkFunction(ExprAST* expr, SymbolTable& st, OptionsTable& opts);

	/// 提取模板信息
	void checkStencilInfo(SymbolTable& st, EquationGroups& eg, TargetFunctionMeta& target,
	                      StencilInfo & stencil_info);

	/// 提取表达式中的偏移信息
	void checkShiftInfo(const ExprAST* expr, SymbolTable& st, TargetFunctionMeta& target,
	                    StencilInfo& shift_info);

	/// 调整方程顺序（拓扑排序）
	void adjestEquationOrder(EquationGroups& eg, TargetFunctionMeta& target);

	/// 从表达式中提取整数值（支持负数）
	static bool getInteger(ExprAST* expr, int64_t& result);

	/// 从表达式中提取浮点数值（支持负数）
	static bool getFloat(ExprAST* expr, double& result);
};
}

#endif //EZ_COMPILE_SEMANTIC_H
