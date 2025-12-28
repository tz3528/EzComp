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

namespace ezcompile {

// ====== SemanticResult ===============

using SymbolId = uint32_t;

// ------------------------------
// 1) 选项表 OptionsTable
// ------------------------------
struct OptionsTable {
  // 原始 options：保留扩展性（未来新增 option 不破坏结构）
  struct Value {
    enum class Kind { Str, Int, Float } kind;
    std::string s;
    int64_t i = 0;
    double f = 0.0;
  };

  // raw["mode"] = "time-pde" 等
  std::unordered_map<std::string, Value> raw;

  // 下列是 lowering 常用的“已解析字段”（你 examples 中出现过）
  std::string mode;    // 求解模式，目前只支持时序方程求解
  std::string method;  // 求解方法，目前只支持有限差分法
  int precisionExp10 = 0; // e.g. -6 表示 10^(-6) 的精度语义

  // function:"u(x,t)" 解析后的结构（给 lowering 不用再拆字符串）
  struct FunctionSig {
    std::string name;                // "u"
    std::vector<std::string> args;   // ["x","t"]
    std::string text;                // 原始文本 "u(x,t)"，便于 debug/打印
  } targetFunc;

  // timeVar:"t"：用于识别时间维（loop 嵌套/初始化条件识别）
  std::string timeVar;
};

// ------------------------------
// 2) 符号表 SymbolTable
// ------------------------------
// declarations: IDENT "[" INT "," INT "," INT "]" ";"
// 三个参数分别：下界、上界、区间点数
struct Domain {
  int64_t lower = 0.0;   // 下界
  int64_t upper = 0.0;   // 上界
  int64_t points = 0;   // 点数 N

  // 均匀网格步长
  // step = (upper - lower) / (points - 1)  (points>=2)
  double step = 0.0;
};

enum class SymbolKind {
  IndependentVar,   // 来自 declarations 的自变量：x / t ...
  UnknownFunction,  // 来自 options.function 的未知函数：u ...
  // 未来可扩展：Parameter / Constant / Builtin(diff/delta) 等
};

struct Symbol {
  SymbolId id = 0;
  std::string name;
  SymbolKind kind = SymbolKind::IndependentVar;

  // 对 IndependentVar：domain 必填
  std::optional<Domain> domain;

  // 对 UnknownFunction：参数名列表（从 options.function 解析）
  std::vector<std::string> params;
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

// ------------------------------------------------------
// 3/4/5) 方程分组 EquationGroups
// ------------------------------------------------------
struct EquationGroups {
  // 初始化方程列表：如 u(x,0)=0;
  std::vector<const EquationAST*> init;

  // 边界方程列表：如 u(0,t)=10; u(100,t)=10;
  std::vector<const EquationAST*> boundary;

  // 每次迭代的普通方程：如 PDE 主方程/离散模板等
  std::vector<const EquationAST*> iter;
};

// ------------------------------------------------------
// TargetFunctionMeta
// - 目标未知函数符号（u）
// - 时间维符号（t）
// - 空间维符号（x、y…）
struct TargetFunctionMeta {
  SymbolId funcSym = 0;      // u
  SymbolId timeDim = 0;      // t
  std::vector<SymbolId> spaceDims; // [x] 或 [x,y]
};

// ------------------------------------------------------
// 可选但你现在认可：Anchor（用于 init/boundary 落到切片/索引）
// ------------------------------------------------------
// 目的：当你把 PDE lowering 成循环或张量操作时，
//       需要把 u(0,t) 映射为 U[0, :]，把 u(100,t) 映射为 U[N-1, :]。
//       这类信息如果在 sema 阶段算好，中端不用再扫 LHS/对照 declarations 推导一次。
enum class BoundarySide { Min, Max };

// 一个“维度被固定为常量”的锚点信息（只服务 init/boundary，不泛化到所有方程）
struct Anchor {
  SymbolId dim = 0;                 // 被固定的维度
  double value = 0.0;               // 固定值
  std::optional<BoundarySide> side; // 若 value==domain.lower/upper，则分别为 Min/Max
  std::optional<int64_t> index;     // 若能落到均匀网格索引：0 或 N-1（或 t 的 0）
};

// 针对 init/boundary 方程的锚点缓存：
// - init：通常 anchors 里含 {dim=t, value=0, index=0}
// - boundary：通常 anchors 里含 {dim=x, value=0/100, side=Min/Max, index=0/N-1}
struct AnchorTable {
  // 注意：key 用 AST 指针即可（语义分析阶段已经分组并持有指针）
  std::unordered_map<const EquationAST*, std::vector<Anchor>> anchorsOf;
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

  // Optional：init/boundary 的 锚点表
  std::optional<AnchorTable> anchorTable;
};

/// 用于存储解析结果和语法分析中的内容
struct ParsedModule {
	mlir::MLIRContext context;
	llvm::SourceMgr sourceMgr;
	int bufferID = 0;
	std::unique_ptr<ModuleAST> module;
	std::unique_ptr<SemanticResult> sema;
};


class Semantic : public DiagnosticBase {
public:
  Semantic(llvm::SourceMgr &sourceMgr, int bufferID, mlir::MLIRContext *ctx)
    : DiagnosticBase(sourceMgr, bufferID, ctx) {};

  std::unique_ptr<SemanticResult> analyze(const ModuleAST& module);

private:
  void collectDecls(const ModuleAST &module);
  void checkOptions(const ModuleAST &module);
  void checkEquations(const ModuleAST &module);

};

}

#endif //EZ_COMPILE_SEMANTIC_H
