//===-- MLIRGen.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// MLIRGen用于将AST转换为comp方言
//
//===----------------------------------------------------------------------===//


#ifndef EZ_COMPILE_MLIR_GEN_H
#define EZ_COMPILE_MLIR_GEN_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Value.h"

#include "Comp.h"
#include "Semantic/Semantic.h"
#include "Dialects/Comp/include/Comp.h"


namespace mlir {
class OpBuilder;
template <typename OpTy>
class OwningOpRef;
class ModuleOp;
}

namespace ezcompile {

/// MLIR代码生成器：将AST转换为Comp方言的MLIR IR
class MLIRGen {

	struct TimeLoopCtx;
	mlir::FloatType f64Ty;

public:
	MLIRGen(const ParsedModule &pm, mlir::MLIRContext &context);

	mlir::FailureOr<mlir::ModuleOp> mlirGen();

	void print(llvm::raw_ostream &os,
	           mlir::Operation *op = nullptr,
	           mlir::OpPrintingFlags flags = {}) const;

private:
	//===--------------------------------------------------------------------===//
	// 顶层：生成模块/Problem
	//===--------------------------------------------------------------------===//
	/// 生成顶层comp.problem操作
	mlir::FailureOr<comp::ProblemOp> genProblem();

	mlir::LogicalResult genDim();
	mlir::FailureOr<mlir::Value> genField();
	mlir::LogicalResult genSolve(mlir::Value field);
	mlir::LogicalResult genApplyInit(mlir::Value field);
	mlir::LogicalResult genDirichlet(mlir::Value field);
	mlir::LogicalResult genForTime(mlir::Value field, mlir::Value timePoints);
	mlir::LogicalResult genUpdate(mlir::Value field, TimeLoopCtx tctx);
	mlir::LogicalResult genSample(mlir::Value field);

	//===--------------------------------------------------------------------===//
	// Region helper：终结符
	//===--------------------------------------------------------------------===//
	mlir::LogicalResult emitYield(mlir::Location loc, mlir::Value value);

	//===--------------------------------------------------------------------===//
	// 表达式生成
	//===--------------------------------------------------------------------===//
	mlir::FailureOr<mlir::Value> genExpr(const ExprAST * expr);
	mlir::FailureOr<mlir::Value> genIntExpr(const IntExprAST * expr);
	mlir::FailureOr<mlir::Value> genFloatExpr(const FloatExprAST * expr);
	mlir::FailureOr<mlir::Value> genVarRefExpr(const VarRefExprAST * expr);
	mlir::FailureOr<mlir::Value> genUnaryExpr(const UnaryExprAST * expr);
	mlir::FailureOr<mlir::Value> genBinaryExpr(const BinaryExprAST * expr);
	mlir::FailureOr<mlir::Value> genCallExpr(const CallExprAST * expr);
	mlir::FailureOr<mlir::Value> genParenExpr(const ParenExprAST * expr);

	// points/delta 更适合作为表达式层 helper（用到才生成）
	mlir::FailureOr<mlir::Value> genPoints(mlir::Location loc, SymbolId dimId);

	/// 报错接口
	mlir::LogicalResult emitError(llvm::SMLoc loc, llvm::StringRef msg) {
		// loc 无效：只能打印纯文本
		if (!loc.isValid()) {
			llvm::errs() << "error: " << msg << "\n";
			return mlir::failure();
		}

		unsigned bufID = pm.sourceMgr.FindBufferContainingLoc(loc);
		if (bufID == 0) {
			llvm::errs() << "error: " << msg << "\n";
			return mlir::failure();
		}

		pm.sourceMgr.PrintMessage(loc, llvm::SourceMgr::DK_Error, msg);
		return mlir::failure();
	}

	// 根据 id 获取 dim 的符号引用
	mlir::FlatSymbolRefAttr mkDimRef(SymbolId id) {
		static const auto &symtab = pm.sema->st;
		const auto &sym = symtab.get(id);
		return mlir::FlatSymbolRefAttr::get(&context, sym.name);
	}

	const ParsedModule &pm;
	const SemanticResult *sema;
	mlir::MLIRContext &context;
	mlir::OpBuilder builder;
	mlir::ModuleOp IRModule;

	llvm::DenseMap<SymbolId, mlir::Value> dimIndexEnv;  // 维度 -> 索引值
	llvm::DenseMap<SymbolId, mlir::Value> dimCoordEnv;  // 维度 -> 坐标值

	std::map<ShiftInfo, mlir::Value> shiftInfoEnv;  // Stencil偏移 -> sample操作缓存

	llvm::StringMap<mlir::Value> eqValue;			// 根据函数的完整文本找到对应的句柄

	/// 时间循环上下文
	struct TimeLoopCtx {
		comp::ForTimeOp loop;			// 循环操作符
		mlir::Value atTime;				// 当前时刻
		mlir::Value lb, ub, step;		// 寻话初值、循环上界和步长

		static TimeLoopCtx makeTimeLoopCtx(comp::ForTimeOp loop) {
			mlir::Block &entry = loop.getBody().front();
			mlir::Value atTime = entry.getArgument(0);
			return {loop, atTime, loop.getLb(), loop.getUb(), loop.getStep()};
		}
	};
};

}

#endif //EZ_COMPILE_MLIR_GEN_H
