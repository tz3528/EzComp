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
#include "Semantic.h"
#include "Dialects/Comp/include/Comp.h"


namespace mlir {
class OpBuilder;
template <typename OpTy>
class OwningOpRef;
class ModuleOp;
}

namespace ezcompile {

class MLIRGen {
public:
	MLIRGen(const ParsedModule &pm, mlir::MLIRContext &context);

	mlir::FailureOr<mlir::ModuleOp> mlirGen();

	/**
	 *	打印接口，默认输出所有ir
	 * @param os 输出流，通常是llvm::errs()
	 * @param op 要输出的操作，默认是IRModule
	 * @param flags 打印选项
	 */
	void print(llvm::raw_ostream &os,
	           mlir::Operation *op = nullptr,
	           mlir::OpPrintingFlags flags = {}) const;

private:
	//===--------------------------------------------------------------------===//
	// 顶层：生成模块/Problem
	//===--------------------------------------------------------------------===//
	mlir::FailureOr<comp::ProblemOp> genProblem();

	mlir::LogicalResult genDim(comp::ProblemOp problem);
	mlir::FailureOr<mlir::Value> genField(comp::ProblemOp problem);
	mlir::LogicalResult genSolve(comp::ProblemOp problem, mlir::Value field);
	mlir::LogicalResult genApplyInit(comp::ProblemOp problem);
	mlir::LogicalResult genDirichlet(comp::ProblemOp problem);
	mlir::LogicalResult genForTime(comp::ProblemOp problem);
	mlir::LogicalResult genUpdate(comp::ProblemOp problem);
	mlir::LogicalResult genSample(comp::ProblemOp problem);
	mlir::LogicalResult genEnforceBoundary(comp::ProblemOp problem);

	//===--------------------------------------------------------------------===//
	// Region helper：终结符
	//===--------------------------------------------------------------------===//
	mlir::LogicalResult emitYield(mlir::Location loc, mlir::Value value);

	//===--------------------------------------------------------------------===//
	// 表达式生成
	//===--------------------------------------------------------------------===//
	mlir::FailureOr<mlir::Value> genExpr(const ExprAST * expr);
	mlir::FailureOr<mlir::Value> genStringExpr(const StringExprAST * expr);
	mlir::FailureOr<mlir::Value> genIntExpr(const IntExprAST * expr);
	mlir::FailureOr<mlir::Value> genFloatExpr(const FloatExprAST * expr);
	mlir::FailureOr<mlir::Value> genVarRefExpr(const VarRefExprAST * expr);
	mlir::FailureOr<mlir::Value> genUnaryExpr(const UnaryExprAST * expr);
	mlir::FailureOr<mlir::Value> genBinaryExpr(const BinaryExprAST * expr);
	mlir::FailureOr<mlir::Value> genCallExpr(const CallExprAST * expr);
	mlir::FailureOr<mlir::Value> genParenExpr(const ParenExprAST * expr);

	// points/delta 更适合作为表达式层 helper（用到才生成）
	mlir::FailureOr<mlir::Value> genPoints(mlir::Location loc,
										  mlir::Value value,
										  int64_t index);
	mlir::FailureOr<mlir::Value> genDelta(mlir::Location loc,
										 mlir::Value value,
										 int64_t size,
										 int64_t index);

	/// 报错接口
	void emitError(llvm::SMLoc loc, llvm::StringRef msg) {
		// loc 无效：只能打印纯文本
		if (!loc.isValid()) {
			llvm::errs() << "error: " << msg << "\n";
			return;
		}

		unsigned bufID = pm.sourceMgr.FindBufferContainingLoc(loc);
		if (bufID == 0) {
			llvm::errs() << "error: " << msg << "\n";
			return;
		}

		pm.sourceMgr.PrintMessage(loc, llvm::SourceMgr::DK_Error, msg);
	}

	const ParsedModule &pm;
	mlir::MLIRContext &context;
	mlir::OpBuilder builder;
	mlir::ModuleOp IRModule;
};

}

#endif //EZ_COMPILE_MLIR_GEN_H
