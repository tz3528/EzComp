//===-- MLIRGen.cpp --------------------------------------------*- C++ -*-===//
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

#include "IRGen/include/MLIRGen.h"

#include "mlir/IR/BuiltinOps.h"

namespace ezcompile {

MLIRGen::MLIRGen(const ParsedModule &pm, mlir::MLIRContext &context)
	: pm(pm), builder(&context), context(context) {

}

mlir::FailureOr<mlir::ModuleOp> MLIRGen::mlirGen() {
	// 初始化 IRModule
	mlir::Location loc = mlir::UnknownLoc::get(&context);
	IRModule = mlir::ModuleOp::create(loc);

	// 让 builder 从 module body 开头插入
	mlir::OpBuilder::InsertionGuard guard(builder);
	builder.setInsertionPointToStart(IRModule.getBody());

	// 生成顶层 Problem
	auto problemOr = genProblem();
	if (mlir::failed(problemOr))
		return mlir::failure();

	// 检查IRModule的约束
	if (mlir::failed(mlir::verify(IRModule))) {
		IRModule.emitError() << "module verification failed";
		return mlir::failure();
	}

	return IRModule;
}

void MLIRGen::print(llvm::raw_ostream &os,
		   mlir::Operation *op,
		   mlir::OpPrintingFlags flags) const {

	// 默认打印所有的ir
	if (!op) {
		if (!IRModule) {
			os << "<<MLIRGen: no module to print>>\n";
			return;
		}
		mlir::ModuleOp module = IRModule;
		op = module.getOperation(); //此操作是非const的，所以需要拷贝一份
	}

	// 使SSA命名更加简洁且本地化
	flags.useLocalScope();

	op->print(os, flags);
	os << '\n';
}

mlir::FailureOr<comp::ProblemOp> MLIRGen::genProblem() {
	mlir::Location loc = mlir::UnknownLoc::get(&context);

	// -------- 0) 基本前置条件检查（避免空指针）--------
	if (!pm.sema) {
		emitError(llvm::SMLoc(), "internal error: missing SemanticResult");
		return mlir::failure();
	}
	if (!pm.sema->target.has_value()) {
		emitError(llvm::SMLoc(), "semantic error: missing target function meta");
		return mlir::failure();
	}

	// -------- 1) 在 module 顶层创建 comp.problem --------
	auto problem = comp::ProblemOp::create(builder, loc);

	// 枚举选项表的内容作为problem的属性
	const auto &opts = pm.sema->options;
	for (const auto & opt : opts) {
		auto key = opt.getKey().str();
		auto value = opt.getValue().value;

		if (const auto* s = std::get_if<std::string>(&value)) {
			problem->setAttr(key, builder.getStringAttr(*s));
		}
		else if (const auto* x = std::get_if<int64_t>(&value)) {
			problem->setAttr(key, builder.getI64IntegerAttr(*x));
		}
		else if (const auto* d = std::get_if<double>(&value)) {
			problem->setAttr(key, builder.getF64FloatAttr(*d));
		}
	}

	// comp.problem 有一个 body region
	mlir::Region &body = problem.getBody();
	if (body.empty()) {
		body.emplaceBlock();
	}

	// 之后的 dim/field/solve 都插到 problem body 里
	mlir::OpBuilder::InsertionGuard guard(builder);
	builder.setInsertionPointToStart(&body.front());

	// -------- 2) 生成 dims --------
	if (mlir::failed(genDim(problem))) {
		return mlir::failure();
	}

	// -------- 3) 生成 field --------
	mlir::FailureOr<mlir::Value> fieldOr = genField(problem);
	if (mlir::failed(fieldOr)) {
		return mlir::failure();
	}
	mlir::Value field = *fieldOr;

	// -------- 4) 生成 solve --------
	if (mlir::failed(genSolve(problem, field))) {
		return mlir::failure();
	}

	return problem;
}

mlir::LogicalResult MLIRGen::genDim(comp::ProblemOp problem) {
	mlir::Location loc = mlir::UnknownLoc::get(&context);

	const auto &symbols = pm.sema->st.symbols;
	for (const auto &sym : symbols) {
		auto nameAttr = builder.getStringAttr(sym.name);
		auto lowerAttr  = builder.getF64FloatAttr(static_cast<double>(sym.domain.lower));
		auto upperAttr  = builder.getF64FloatAttr(static_cast<double>(sym.domain.upper));
		auto pointsAttr = builder.getI64IntegerAttr(static_cast<int64_t>(sym.domain.points));

		comp::DimOp::create(builder,loc, nameAttr, lowerAttr, upperAttr, pointsAttr);
	}

	return mlir::success();
}

mlir::FailureOr<mlir::Value> MLIRGen::genField(comp::ProblemOp problem) {
	mlir::Location loc = mlir::UnknownLoc::get(&context);

	auto target = *pm.sema->target;
	std::string fieldName = target.func;

	const auto &symtab = pm.sema->st;

	// 根据 id 获取 dim 的符号引用
	auto mkDimRef = [&](SymbolId id) -> mlir::FlatSymbolRefAttr {
		const auto &sym = symtab.get(id);
		return mlir::FlatSymbolRefAttr::get(&context, sym.name);
	};

	llvm::SmallVector<mlir::Attribute, 4> spaceDimRefs;
	spaceDimRefs.reserve(target.spaceDims.size());
	for (auto sd : target.spaceDims) {
		spaceDimRefs.emplace_back(mkDimRef(sd));
	}
	mlir::ArrayAttr spaceDimsAttr = builder.getArrayAttr(spaceDimRefs);

	mlir::FlatSymbolRefAttr timeDimAttr = mkDimRef(target.timeDim);
	mlir::Type fieldTy = comp::FieldType::get(&context, builder.getF64Type());

	// 创建 comp.field，并拿到它的 SSA 结果 Value
	auto fieldOp = comp::FieldOp::create(
		builder, loc, fieldTy, builder.getStringAttr(fieldName),
		spaceDimsAttr, timeDimAttr
	);

	return fieldOp.getResult();
}

mlir::LogicalResult MLIRGen::genSolve(comp::ProblemOp problem, mlir::Value field) {

	return mlir::success();
}

mlir::LogicalResult MLIRGen::genApplyInit(comp::ProblemOp problem) {

}

mlir::LogicalResult MLIRGen::genDirichlet(comp::ProblemOp problem) {

}

mlir::LogicalResult MLIRGen::genForTime(comp::ProblemOp problem) {

}

mlir::LogicalResult MLIRGen::genUpdate(comp::ProblemOp problem) {

}

mlir::LogicalResult MLIRGen::genSample(comp::ProblemOp problem) {

}

mlir::LogicalResult MLIRGen::genEnforceBoundary(comp::ProblemOp problem) {

}

mlir::LogicalResult MLIRGen::emitYield(mlir::Location loc, mlir::Value value) {

}

mlir::FailureOr<mlir::Value> MLIRGen::genExpr(const ExprAST * expr) {

}

mlir::FailureOr<mlir::Value> MLIRGen::genStringExpr(const StringExprAST * expr) {

}

mlir::FailureOr<mlir::Value> MLIRGen::genIntExpr(const IntExprAST * expr) {

}

mlir::FailureOr<mlir::Value> MLIRGen::genFloatExpr(const FloatExprAST * expr) {

}

mlir::FailureOr<mlir::Value> MLIRGen::genVarRefExpr(const VarRefExprAST * expr) {

}

mlir::FailureOr<mlir::Value> MLIRGen::genUnaryExpr(const UnaryExprAST * expr) {

}

mlir::FailureOr<mlir::Value> MLIRGen::genBinaryExpr(const BinaryExprAST * expr) {

}

mlir::FailureOr<mlir::Value> MLIRGen::genCallExpr(const CallExprAST * expr) {

}

mlir::FailureOr<mlir::Value> MLIRGen::genParenExpr(const ParenExprAST * expr) {

}

mlir::FailureOr<mlir::Value> MLIRGen::genPoints(
	mlir::Location loc, mlir::Value value, int64_t index) {

}
mlir::FailureOr<mlir::Value> MLIRGen::genDelta(
	mlir::Location loc, mlir::Value value, int64_t size, int64_t index) {

}


}
