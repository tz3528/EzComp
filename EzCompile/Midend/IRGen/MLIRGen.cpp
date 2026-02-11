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

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "MLIRGen.h"

namespace ezcompile {
MLIRGen::MLIRGen(const ParsedModule& pm, mlir::MLIRContext& context)
	: pm(pm), builder(&context), context(context) {
	sema = pm.sema.get();
	f64Ty = builder.getF64Type();
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

void MLIRGen::print(llvm::raw_ostream& os,
                    mlir::Operation* op,
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
		return mlir::emitError(loc, "internal error: missing SemanticResult");
	}

	// -------- 1) 在 module 顶层创建 comp.problem --------
	auto problem = comp::ProblemOp::create(builder, loc);

	// 枚举选项表的内容作为problem的属性
	const auto& opts = pm.sema->options;
	for (const auto& opt : opts) {
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
	mlir::Region& body = problem.getBody();
	if (body.empty()) {
		body.emplaceBlock();
	}

	// 之后的 dim/field/solve 都插到 problem body 里
	mlir::OpBuilder::InsertionGuard guard(builder);
	builder.setInsertionPointToStart(&body.front());

	// -------- 2) 生成 dims --------
	if (mlir::failed(genDim())) {
		return mlir::failure();
	}

	// -------- 3) 生成 field --------
	mlir::FailureOr<mlir::Value> fieldOr = genField();
	if (mlir::failed(fieldOr)) {
		return mlir::failure();
	}
	mlir::Value field = *fieldOr;

	// -------- 4) 生成 solve --------
	if (mlir::failed(genSolve(field))) {
		return mlir::failure();
	}

	return problem;
}

mlir::LogicalResult MLIRGen::genDim() {
	mlir::Location loc = mlir::UnknownLoc::get(&context);

	{
		auto sym = sema->st.get(sema->target.timeDim);
		auto nameAttr = builder.getStringAttr(sym.name);
		auto lowerAttr = builder.getF64FloatAttr(static_cast<double>(sym.domain.lower));
		auto upperAttr = builder.getF64FloatAttr(static_cast<double>(sym.domain.upper));
		auto pointsAttr = builder.getI64IntegerAttr(static_cast<int64_t>(sym.domain.points));
		auto timeVarAttr = builder.getUnitAttr();
		comp::DimOp::create(builder, loc, nameAttr, lowerAttr, upperAttr, pointsAttr, timeVarAttr);
	}

	for (const auto &id : sema->target.spaceDims) {
		auto sym = sema->st.get(id);
		auto nameAttr = builder.getStringAttr(sym.name);
		auto lowerAttr = builder.getF64FloatAttr(static_cast<double>(sym.domain.lower));
		auto upperAttr = builder.getF64FloatAttr(static_cast<double>(sym.domain.upper));
		auto pointsAttr = builder.getI64IntegerAttr(static_cast<int64_t>(sym.domain.points));
		comp::DimOp::create(builder, loc, nameAttr, lowerAttr, upperAttr, pointsAttr);
	}

	return mlir::success();
}

mlir::FailureOr<mlir::Value> MLIRGen::genField() {
	mlir::Location loc = mlir::UnknownLoc::get(&context);

	const auto& target = pm.sema->target;
	std::string fieldName = target.func;

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

mlir::LogicalResult MLIRGen::genSolve(mlir::Value field) {
	mlir::Location loc = mlir::UnknownLoc::get(&context);

	auto solve = comp::SolveOp::create(builder, loc, field);

	// Init region
	{
		mlir::Region& initRegion = solve.getInit();
		initRegion.emplaceBlock();

		mlir::OpBuilder::InsertionGuard guard(builder);
		builder.setInsertionPointToStart(&initRegion.front());

		if (mlir::failed(genApplyInit(field))) {
			return mlir::failure();
		}
	}

	// Boundary region
	{
		mlir::Region& boundaryRegion = solve.getBoundary();
		boundaryRegion.emplaceBlock();

		mlir::OpBuilder::InsertionGuard guard(builder);
		builder.setInsertionPointToStart(&boundaryRegion.front());

		if (mlir::failed(genDirichlet(field))) {
			return mlir::failure();
		}
	}

	// Step region
	{
		mlir::Region& stepRegion = solve.getStep();
		stepRegion.emplaceBlock();

		mlir::OpBuilder::InsertionGuard guard(builder);
		builder.setInsertionPointToStart(&stepRegion.front());

		auto timePoints = genPoints(loc, pm.sema->target.timeDim);
		if (mlir::failed(timePoints)) {
			return mlir::failure();
		}

		if (mlir::failed(genForTime(field, *timePoints))) {
			return mlir::failure();
		}
	}

	return mlir::success();
}

mlir::LogicalResult MLIRGen::genApplyInit(mlir::Value field) {
	const auto& meta = pm.sema->target;

	auto initEa = pm.sema->egs.init;
	auto idxTy = builder.getIndexType();

	for (const auto& ea : initEa) {
		mlir::Location loc = mlir::UnknownLoc::get(&context);

		const auto& anchor = ea.anchor;

		// 构建 anchors 属性
		llvm::SmallVector<mlir::Attribute, 4> anchors;
		llvm::SmallDenseSet<SymbolId, 8> fixedDims; // 获取所有被固定的维度

		for (size_t i = 0; i < anchor.dim.size(); ++i) {
			comp::AnchorAttr aa = comp::AnchorAttr::get(
				&context,
				mkDimRef(anchor.dim[i]),
				anchor.index[i]
			);
			anchors.emplace_back(aa);
			fixedDims.insert(anchor.dim[i]);
		}

		if (!fixedDims.contains(meta.timeDim)) {
			return mlir::emitError(loc, "In the initialization equation, timeVar must be fixed");
		}

		auto anchorsAttr = builder.getArrayAttr(anchors);

		// 创建 comp.apply_init 操作
		auto aiOp = comp::ApplyInitOp::create(builder, loc, field, anchorsAttr);

		// 获取并设置 region
		mlir::Region& aiRegion = aiOp.getRegion();
		mlir::Block& entry = aiRegion.emplaceBlock();

		llvm::SmallVector<SymbolId, 4> freeDims;
		for (SymbolId d : meta.spaceDims) {
			if (!fixedDims.contains(d)) freeDims.push_back(d);
		}

		// 没被固定的维度需要嵌套循环枚举
		llvm::SmallVector<mlir::Value, 4> freeIdxArgs;
		for (size_t k = 0; k < freeDims.size(); ++k) {
			freeIdxArgs.push_back(entry.addArgument(idxTy, loc));
		}

		mlir::OpBuilder::InsertionGuard guard(builder);
		builder.setInsertionPointToStart(&entry);

		dimIndexEnv.clear();
		dimCoordEnv.clear();

		// 自由维度：索引来自 block argument
		for (size_t k = 0; k < freeDims.size(); ++k) {
			SymbolId d = freeDims[k];
			dimIndexEnv[d] = freeIdxArgs[k];
			dimCoordEnv[d] = comp::CoordOp::create(builder, loc, f64Ty, mkDimRef(d), freeIdxArgs[k]);
		}

		// 固定维度：索引来自 anchors（常量）
		for (size_t i = 0; i < anchor.dim.size(); ++i) {
			SymbolId d = anchor.dim[i];
			uint64_t ix = anchor.index[i];
			mlir::Value cix = mlir::arith::ConstantIndexOp::create(builder, loc, ix);
			dimIndexEnv[d] = cix;
			dimCoordEnv[d] = comp::CoordOp::create(builder, loc, f64Ty, mkDimRef(d), cix);
		}

		// 生成 RHS（边界值表达式）
		auto valueOr = genExpr(ea.eq->getRHS());
		if (mlir::failed(valueOr)) return mlir::failure();
		if (mlir::failed(emitYield(loc, *valueOr))) return mlir::failure();
	}

	return mlir::success();
}

mlir::LogicalResult MLIRGen::genDirichlet(mlir::Value field) {
	mlir::Location loc = mlir::UnknownLoc::get(&context);
	const auto& meta = pm.sema->target;

	auto boundaryEa = pm.sema->egs.boundary;
	auto idxTy = builder.getIndexType();

	for (const auto& ea : boundaryEa) {
		const auto& anchor = ea.anchor;

		llvm::SmallVector<mlir::Attribute, 4> anchors;
		llvm::SmallDenseSet<SymbolId, 8> fixedDims;

		for (size_t i = 0; i < anchor.dim.size(); ++i) {
			comp::AnchorAttr aa = comp::AnchorAttr::get(
				&context,
				mkDimRef(anchor.dim[i]),
				anchor.index[i]
			);
			anchors.emplace_back(aa);
			fixedDims.insert(anchor.dim[i]);
		}

		auto anchorsAttr = builder.getArrayAttr(anchors);

		auto dOp = comp::DirichletOp::create(builder, loc, field, anchorsAttr);

		// 获取并设置 region
		mlir::Region& dRegion = dOp.getRegion();
		mlir::Block& entry = dRegion.emplaceBlock();

		llvm::SmallVector<SymbolId, 4> freeDims;
		freeDims.push_back(meta.timeDim);
		for (SymbolId d : meta.spaceDims) {
			if (!fixedDims.contains(d)) freeDims.push_back(d);
		}

		llvm::SmallVector<mlir::Value, 4> freeIdxArgs;
		freeIdxArgs.reserve(freeDims.size());
		for (size_t k = 0; k < freeDims.size(); ++k) {
			freeIdxArgs.push_back(entry.addArgument(idxTy, loc));
		}

		mlir::OpBuilder::InsertionGuard guard(builder);
		builder.setInsertionPointToStart(&entry);

		dimIndexEnv.clear();
		dimCoordEnv.clear();

		// 自由维度：索引来自 block argument
		for (size_t k = 0; k < freeDims.size(); ++k) {
			SymbolId d = freeDims[k];
			mlir::Value ix = freeIdxArgs[k];
			dimIndexEnv[d] = ix;
			dimCoordEnv[d] = comp::CoordOp::create(builder, loc, f64Ty, mkDimRef(d), ix);
		}

		// 固定维度：索引来自 anchors（常量）
		for (size_t i = 0; i < anchor.dim.size(); ++i) {
			SymbolId d = anchor.dim[i];
			int64_t ix = static_cast<int64_t>(anchor.index[i]);
			mlir::Value cix = mlir::arith::ConstantIndexOp::create(builder, loc, ix);
			dimIndexEnv[d] = cix;
			dimCoordEnv[d] = comp::CoordOp::create(builder, loc, f64Ty, mkDimRef(d), cix);
		}

		// 生成 RHS（边界值表达式）
		auto valueOr = genExpr(ea.eq->getRHS());
		if (mlir::failed(valueOr)) return mlir::failure();
		if (mlir::failed(emitYield(loc, *valueOr))) return mlir::failure();
	}

	return mlir::success();
}

mlir::LogicalResult MLIRGen::genForTime(mlir::Value field, mlir::Value timePoints) {
	mlir::Location loc = mlir::UnknownLoc::get(&context);

	mlir::Value c0 = mlir::arith::ConstantIndexOp::create(builder, loc, 0);
	mlir::Value c1 = mlir::arith::ConstantIndexOp::create(builder, loc, 1);

	mlir::Value ub = mlir::arith::SubIOp::create(builder, loc, timePoints, c1);

	auto forOp = comp::ForTimeOp::create(builder, loc, c0, ub, c1);

	mlir::Region& r = forOp.getBody();
	auto* body = new mlir::Block();
	r.push_back(body);
	body->addArgument(builder.getIndexType(), loc);
	auto tctx = TimeLoopCtx::makeTimeLoopCtx(forOp);

	{
		mlir::OpBuilder::InsertionGuard guard(builder);
		builder.setInsertionPointToStart(body);

		tctx.writeTime = mlir::arith::AddIOp::create(builder, loc, tctx.atTime, c1);

		dimIndexEnv.clear();
		dimCoordEnv.clear();
		auto tid = sema->target.timeDim;
		dimIndexEnv[tid] = body->getArgument(0);
		dimCoordEnv[tid] = comp::CoordOp::create(builder, loc, f64Ty, mkDimRef(tid), body->getArgument(0));

		if (mlir::failed(genUpdate(field, tctx))) {
			return mlir::failure();
		}
	}

	return mlir::success();
}

mlir::LogicalResult MLIRGen::genUpdate(mlir::Value field, TimeLoopCtx tctx) {
	mlir::Location loc = mlir::UnknownLoc::get(&context);

	mlir::OpBuilder::InsertionGuard guard(builder);

	llvm::SmallVector<mlir::Attribute, 4> ranges;
	for (SymbolId dimId : pm.sema->target.spaceDims) {
		mlir::FlatSymbolRefAttr dimRef = mkDimRef(dimId);

		int64_t lower = 0, upper = 0;
		for (auto& offset : pm.sema->stencil_info.symbol_info[dimId]) {
			lower = std::min(lower, offset);
			upper = std::max(upper, offset);
		}

		mlir::IntegerAttr lowerAttr = builder.getI64IntegerAttr(lower);
		mlir::IntegerAttr upperAttr = builder.getI64IntegerAttr(upper);
		auto rangeAttr = comp::RangeAttr::get(&context, dimRef, lowerAttr, upperAttr);
		ranges.push_back(rangeAttr);
	}
	mlir::ArrayAttr overAttr = builder.getArrayAttr(ranges);

	auto updateOp = comp::UpdateOp::create(builder, loc, field, tctx.atTime, tctx.writeTime, overAttr);

	mlir::Region& body = updateOp.getBody();
	auto* bb = new mlir::Block();
	body.push_back(bb);

	builder.setInsertionPointToStart(bb);

	// 这里存起来每个变量下标和值的句柄
	for (size_t i = 0; i < sema->target.spaceDims.size(); ++i) {
		bb->addArgument(builder.getIndexType(), loc);
		auto id = sema->target.spaceDims[i];
		dimIndexEnv[id] = bb->getArgument(i);
		dimCoordEnv[id] = comp::CoordOp::create(builder, loc, f64Ty, mkDimRef(id), bb->getArgument(i));
	}

	// sample
	if (mlir::failed(genSample(field))) {
		return mlir::failure();
	}


	// expr
	mlir::Value ans;
	for (auto eq : sema->egs.iter) {
		const ExprAST *lhs = eq->getLHS();
		const ExprAST *rhs = eq->getRHS();

		auto rhsVOr = genExpr(rhs);
		if (mlir::failed(rhsVOr)) {
			return mlir::failure();
		}

		eqValue[lhs->getSourceText()] = *rhsVOr;
		ans = *rhsVOr;
	}

	if (mlir::failed(emitYield(loc, ans))) {
		return mlir::failure();
	}

	return mlir::success();
}

mlir::LogicalResult MLIRGen::genSample(mlir::Value field) {
	mlir::Location loc = mlir::UnknownLoc::get(&context);

	// 这里考虑后面可能有多阶段求解，Sample会多次复用
	shiftInfoEnv.clear();

	auto fieldTy = mlir::dyn_cast<comp::FieldType>(field.getType());
	if (!fieldTy) {
		return mlir::emitError(loc, "fieldTy is null");
	}
	mlir::Type elemTy = fieldTy.getElementType();

	for (auto& info : sema->stencil_info.shift_infos) {
		llvm::SmallVector<mlir::Value, 4> indices;
		llvm::SmallVector<mlir::Attribute, 4> dims;
		llvm::SmallVector<int64_t, 4> shifts;

		for (size_t i = 0; i < info.dim.size(); ++i) {
			auto id = info.dim[i];

			if (dimIndexEnv.find(id) == dimIndexEnv.end()) {
				return mlir::emitError(loc, "The " + std::to_string(i) + " th ID cannot be found in dimIndexEnv");
			}
			auto value = dimIndexEnv[id];
			indices.emplace_back(value);
			dims.emplace_back(mkDimRef(id));
			shifts.emplace_back(info.offset[i]);
		}

		auto dimsAttr = builder.getArrayAttr(dims);
		auto shiftAttr = mlir::DenseI64ArrayAttr::get(builder.getContext(), shifts);

		auto sample = comp::SampleOp::create(builder, loc, elemTy, field, indices, dimsAttr, shiftAttr);

		shiftInfoEnv[info] = sample.getResult();
	}

	return mlir::success();
}

mlir::LogicalResult MLIRGen::emitYield(mlir::Location loc, mlir::Value value) {
	if (!value) {
		return mlir::emitError(loc, "value must be non-null");
	}

	comp::YieldOp::create(builder, loc, value);

	return mlir::success();
}

mlir::FailureOr<mlir::Value> MLIRGen::genExpr(const ExprAST* expr) {
	if (!expr) return mlir::failure();

	if (auto *e = dynamic_cast<const IntExprAST *>(expr)) return genIntExpr(e);
	if (auto *e = dynamic_cast<const FloatExprAST *>(expr)) return genFloatExpr(e);
	if (auto *e = dynamic_cast<const VarRefExprAST *>(expr)) return genVarRefExpr(e);
	if (auto *e = dynamic_cast<const UnaryExprAST *>(expr)) return genUnaryExpr(e);
	if (auto *e = dynamic_cast<const BinaryExprAST *>(expr)) return genBinaryExpr(e);
	if (auto *e = dynamic_cast<const CallExprAST *>(expr)) return genCallExpr(e);
	if (auto *e = dynamic_cast<const ParenExprAST *>(expr)) return genParenExpr(e);

	return emitError(expr->getBeginLoc(), "unsupported expression kind");
}

mlir::FailureOr<mlir::Value> MLIRGen::genIntExpr(const IntExprAST* expr) {
	mlir::Location loc = mlir::UnknownLoc::get(&context);

	const int64_t value = expr->getValue();
	auto i64Ty = builder.getI64Type();
	auto attr = builder.getI64IntegerAttr(value);
	return mlir::arith::ConstantOp::create(builder, loc, i64Ty, attr).getResult();
}

mlir::FailureOr<mlir::Value> MLIRGen::genFloatExpr(const FloatExprAST* expr) {
	mlir::Location loc = mlir::UnknownLoc::get(&context);

	const double value = expr->getValue();
	auto attr = builder.getF64FloatAttr(value);
	return mlir::arith::ConstantOp::create(builder, loc, f64Ty, attr).getResult();
}

mlir::FailureOr<mlir::Value> MLIRGen::genVarRefExpr(const VarRefExprAST* expr) {
	mlir::Location loc = mlir::UnknownLoc::get(&context);
	auto id = sema->st.lookup(expr->getName().str())->id;

	auto it = dimCoordEnv.find(id);
	if (it == dimCoordEnv.end()) {
		return mlir::emitError(loc, expr->getName().str() + " not have a corresponding comp.coord");
	}
	return dimCoordEnv[id];
}

mlir::FailureOr<mlir::Value> MLIRGen::genUnaryExpr(const UnaryExprAST* expr) {
	mlir::Location loc = mlir::UnknownLoc::get(&context);

	auto operand = genExpr(expr->getOperand());
	if (mlir::failed(operand)) {
		return mlir::failure();
	}
	mlir::Value value = *operand;

	char op = expr->getOp();

	if (op == '+') {
		return value;
	}

	if (op == '-') {
		mlir::Type ty = value.getType();

		if (llvm::isa<mlir::FloatType>(ty)) {
			return mlir::arith::NegFOp::create(builder, loc, value).getResult();
		}

		if (llvm::isa<mlir::IntegerType>(ty)) {
			auto zero = mlir::arith::ConstantOp::create(builder, loc, ty,
			                                            builder.getIntegerAttr(ty, 0)).getResult();
			return mlir::arith::SubIOp::create(builder, loc, zero, value).getResult();
		}

		if (ty.isIndex()) {
			auto zero = mlir::arith::ConstantIndexOp::create(builder, loc, 0).getResult();
			return mlir::arith::SubIOp::create(builder, loc, zero, value).getResult();
		}

		return emitError(expr->getBeginLoc(), "only unary '+' and '-' are allowed");
	}

	return emitError(expr->getBeginLoc(), "only unary '+' and '-' are allowed");
}

mlir::FailureOr<mlir::Value> MLIRGen::genBinaryExpr(const BinaryExprAST* expr) {
	mlir::Location loc = mlir::UnknownLoc::get(&context);

	auto lhsOr = genExpr(expr->getLHS());
	if (mlir::failed(lhsOr)) return mlir::failure();
	auto rhsOr = genExpr(expr->getRHS());
	if (mlir::failed(rhsOr)) return mlir::failure();

	mlir::Value lhs = *lhsOr;
	mlir::Value rhs = *rhsOr;
	auto lhsTy = lhs.getType();
	auto rhsTy = rhs.getType();

	auto isInt = [](mlir::Type t) {
		return llvm::isa<mlir::IntegerType>(t) || llvm::isa<mlir::IndexType>(t);
	};

	auto isFloat = [](mlir::Type t) {
		return llvm::isa<mlir::FloatType>(t);
	};

	// 两侧存在浮点数，要将Int转换为浮点数
	if (isFloat(lhsTy) || isFloat(rhsTy)) {
		mlir::FloatType fTy = (isFloat(lhsTy)) ? llvm::cast<mlir::FloatType>(lhsTy) : llvm::cast<mlir::FloatType>(rhsTy);

		auto toFloat = [&](mlir::Value v) -> mlir::Value {
			mlir::Type t = v.getType();
			if (t == fTy) return v;

			if (auto ft = llvm::dyn_cast<mlir::FloatType>(t)) {
				if (ft.getWidth() < fTy.getWidth()) {
					return mlir::arith::ExtFOp::create(builder, loc, fTy, v);
				}
				return mlir::arith::TruncFOp::create(builder, loc, fTy, v);
			}

			if (llvm::isa<mlir::IndexType>(t)) {
				// index -> i64 -> float
				auto i64 = builder.getI64Type();
				auto acOp = mlir::arith::IndexCastOp::create(builder, loc, i64, v);
				return mlir::arith::SIToFPOp::create(builder, loc, fTy, acOp);
			}

			if (auto it = llvm::dyn_cast<mlir::IntegerType>(t)) {
				return mlir::arith::SIToFPOp::create(builder, loc, fTy, v);
			}

			return mlir::Value();
		};

		lhs = toFloat(lhs);
		rhs = toFloat(rhs);
		if (!lhs || !rhs) {
			return mlir::emitError(loc, "lhs and rhs must be non-null");
		}

		switch (expr->getOp()) {
		case '+': return mlir::arith::AddFOp::create(builder, loc, lhs, rhs).getResult();
		case '-': return mlir::arith::SubFOp::create(builder, loc, lhs, rhs).getResult();
		case '*': return mlir::arith::MulFOp::create(builder, loc, lhs, rhs).getResult();
		case '/': return mlir::arith::DivFOp::create(builder, loc, lhs, rhs).getResult();
		default:
			return emitError(expr->getBeginLoc(), std::string(1, expr->getOp()) + " is unexpected");
		}
	}

	// 此时两边应为整型，否则不是数值类型
	if (!isInt(lhsTy) || !isInt(rhsTy)) {
		return emitError(expr->getBeginLoc(), "binary op: operands must be numeric");
	}

	// 到此两端都为整型
	if (lhsTy != rhsTy) {
		if (llvm::isa<mlir::IndexType>(lhsTy) && llvm::isa<mlir::IntegerType>(rhsTy)) {
			rhs = mlir::arith::IndexCastOp::create(builder, loc, lhsTy, rhs);
			rhsTy = rhs.getType();
		}
		else if (llvm::isa<mlir::IndexType>(rhsTy) && llvm::isa<mlir::IntegerType>(lhsTy)) {
			lhs = mlir::arith::IndexCastOp::create(builder, loc, rhsTy, lhs);
			lhsTy = lhs.getType();
		}
		else {
			// 暂时认为整型只有i64一种，不存在其它int类型
			return emitError(expr->getBeginLoc(), "binary op: mismatched integer types");
		}
	}

	switch (expr->getOp()) {
	case '+': return mlir::arith::AddIOp::create(builder, loc, lhs, rhs).getResult();
	case '-': return mlir::arith::SubIOp::create(builder, loc, lhs, rhs).getResult();
	case '*': return mlir::arith::MulIOp::create(builder, loc, lhs, rhs).getResult();
	case '/': return mlir::arith::DivSIOp::create(builder, loc, lhs, rhs).getResult();
	case '%': return mlir::arith::RemSIOp::create(builder, loc, lhs, rhs).getResult();
	default:
		return emitError(expr->getBeginLoc(), std::string(1, expr->getOp()) + " is unexpected");
	}
}

mlir::FailureOr<mlir::Value> MLIRGen::genCallExpr(const CallExprAST* expr) {
	mlir::Location loc = mlir::UnknownLoc::get(&context);

	auto name = expr->getCallee().str();
	if (name == sema->target.func) {
		auto shift_info = sema->stencil_info.call_info.find(expr)->second;
		return shiftInfoEnv[shift_info];
	}
	else if (eqValue.find(expr->getSourceText()) != eqValue.end()) {
		return eqValue[expr->getSourceText()];
	}

	llvm::SmallVector<mlir::Value, 8> operands;

	for (auto &arg : expr->getArgs()) {
		auto vOr = genExpr(arg.get());
		if (mlir::failed(vOr)) {
			return mlir::failure();
		}

		operands.push_back(*vOr);
	}

	auto calleeName  = expr->getCallee().str();
	auto calleeAttr  = mlir::FlatSymbolRefAttr::get(&context, calleeName);
	mlir::Value callOp = comp::CallOp::create(builder, loc, f64Ty, calleeAttr, operands).getResult();

	return callOp;
}

mlir::FailureOr<mlir::Value> MLIRGen::genParenExpr(const ParenExprAST* expr) {
	return genExpr(expr->getSub());
}

mlir::FailureOr<mlir::Value> MLIRGen::genPoints(mlir::Location loc, SymbolId dimId) {
	mlir::FlatSymbolRefAttr dimRef = mkDimRef(dimId);
	if (!dimRef) {
		return mlir::emitError(loc,
		                       "genPoints: failed to make dim symbol ref for id=" + std::to_string(dimId));
	}

	// 生成：%N = comp.points @t : index
	auto op = comp::PointsOp::create(builder, loc, builder.getIndexType(), dimRef);
	return op.getResult();
}


}
