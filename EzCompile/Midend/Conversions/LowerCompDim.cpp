//===-- LowerCompDim.cpp ----------------------------------------*- C++ -*-===//
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


#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "Dialects/Comp/include/Comp.h"

namespace ezcompile {

static constexpr mlir::StringLiteral kDimsAttrName = "comp.dims";

struct LowerCompDimPass : mlir::PassWrapper<LowerCompDimPass, mlir::OperationPass<mlir::ModuleOp>> {
	MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerCompDimPass)

	void getDependentDialects(mlir::DialectRegistry& registry) const override {
		registry.insert<mlir::arith::ArithDialect, comp::CompDialect>();
	}

	mlir::StringRef getArgument() const override { return "lower-comp-dim"; }

	mlir::StringRef getDescription() const override {
		return "Lower comp.dim: verify + write module metadata + materialize constants";
	}

	void runOnOperation() override {
		mlir::ModuleOp module = getOperation();
		mlir::MLIRContext* ctx = module.getContext();

		auto problems = module.getOps<comp::ProblemOp>();
		auto it = problems.begin();
		if (it == problems.end()) {
			module.emitError() << "expected exactly one comp.problem, got 0";
			signalPassFailure();
			return;
		}
		comp::ProblemOp problem = *it;
		++it;
		if (it != problems.end()) {
			module.emitError() << "expected exactly one comp.problem, got >1";
			signalPassFailure();
			return;
		}

		// 入口插常量：lower/upper/points
		mlir::Block& entry = problem.getBody().front();
		mlir::OpBuilder entryBuilder(&entry, entry.begin());

		// 收集 module 级 dims metadata
		mlir::NamedAttrList dimsKVs;

		bool anyFailure = false;

		// 找到所有dim并降级
		problem.walk([&](comp::DimOp dimOp) {
			mlir::StringRef dimName = dimOp.getSymName();

			mlir::FloatAttr lowerAttr = dimOp.getLowerAttr();
			mlir::FloatAttr upperAttr = dimOp.getUpperAttr();
			mlir::IntegerAttr pointsAttr = dimOp.getPointsAttr();

			int64_t points = pointsAttr.getInt();
			double lower = lowerAttr.getValueAsDouble();
			double upper = upperAttr.getValueAsDouble();

			// 写入 module metadata: comp.dims = { "x" = {lower=..., upper=..., points=...}, ... }
			mlir::DictionaryAttr oneDim = mlir::DictionaryAttr::get(
				ctx, {
					mlir::NamedAttribute(mlir::StringAttr::get(ctx, "lower"), lowerAttr),
					mlir::NamedAttribute(mlir::StringAttr::get(ctx, "upper"), upperAttr),
					mlir::NamedAttribute(mlir::StringAttr::get(ctx, "points"), pointsAttr),
				});

			dimsKVs.set(mlir::StringAttr::get(ctx, dimName), oneDim);

			// 物化常量（插在 comp.problem 入口）
			auto cLower = mlir::arith::ConstantOp::create(entryBuilder, dimOp.getLoc(),
			                                                     lowerAttr.getType(), lowerAttr);
			cLower->setAttr("comp.dim.name", mlir::StringAttr::get(ctx, dimName));
			cLower->setAttr("comp.dim.kind", mlir::StringAttr::get(ctx, "lower"));

			auto cUpper = mlir::arith::ConstantOp::create(entryBuilder, dimOp.getLoc(),
			                                                     upperAttr.getType(), upperAttr);
			cUpper->setAttr("comp.dim.name", mlir::StringAttr::get(ctx, dimName));
			cUpper->setAttr("comp.dim.kind", mlir::StringAttr::get(ctx, "upper"));

			auto cPts = mlir::arith::ConstantIndexOp::create(entryBuilder, dimOp.getLoc(), points);
			cPts->setAttr("comp.dim.name", mlir::StringAttr::get(ctx, dimName));
			cPts->setAttr("comp.dim.kind", mlir::StringAttr::get(ctx, "points"));

			// 先不 erase dim，给后续 field/solve pass 留着（最后统一 cleanup）
			dimOp->setAttr("comp.lowered", mlir::UnitAttr::get(ctx));
		});

		if (anyFailure) {
			return;
		}

		module->setAttr(kDimsAttrName, mlir::DictionaryAttr::get(ctx, dimsKVs));
	}
};

void registerLowerCompDimPass() {
	mlir::PassRegistration<LowerCompDimPass>();
}

std::unique_ptr<mlir::Pass> createLowerCompDimPass() {
	return std::make_unique<LowerCompDimPass>();
}

}
