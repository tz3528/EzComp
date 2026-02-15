//===-- LowerUtil.h --------------------------------------------*- C++ -*-===//
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


#ifndef EZ_COMPILE_LOWER_UTIL_H
#define EZ_COMPILE_LOWER_UTIL_H

namespace ezcompile {

// 附近通过符号引用查找 comp.dim
inline comp::DimOp lookupDimOp(mlir::Operation* from, mlir::FlatSymbolRefAttr dimSym) {
	if (!dimSym) return {};
	mlir::Operation* sym = mlir::SymbolTable::lookupNearestSymbolFrom(from, dimSym.getAttr());
	return dyn_cast_or_null<comp::DimOp>(sym);
}

// coord操作降级，计算方式为 lb + (ub - lb) * (index / (points - 1))
inline mlir::Value lowerCoord(mlir::OpBuilder& b, mlir::Location loc, mlir::Operation* anchorOp,
							  mlir::FlatSymbolRefAttr dimSym, mlir::Value ivIndex) {
	comp::DimOp dimOp = lookupDimOp(anchorOp, dimSym);
	if (!dimOp) {
		anchorOp->emitError() << "cannot resolve comp.dim for " << dimSym;
		return {};
	}

	double lower = dimOp.getLower().convertToDouble();
	double upper = dimOp.getUpper().convertToDouble();
	int64_t points = static_cast<int64_t>(dimOp.getPoints());

	mlir::Type f64 = b.getF64Type();
	mlir::Type i64 = b.getI64Type();

	mlir::Value cLower = mlir::arith::ConstantFloatOp::create(b, loc, cast<mlir::FloatType>(f64), mlir::APFloat(lower));
	mlir::Value cUpper = mlir::arith::ConstantFloatOp::create(b, loc, cast<mlir::FloatType>(f64), mlir::APFloat(upper));

	mlir::Value cPointsI64 = mlir::arith::ConstantIntOp::create(b, loc, i64, points);
	mlir::Value cOneI64 = mlir::arith::ConstantIntOp::create(b, loc, i64, 1);
	mlir::Value denomI64 = mlir::arith::SubIOp::create(b, loc, cPointsI64, cOneI64);

	// iv: index -> i64 -> f64
	mlir::Value ivI64 = mlir::arith::IndexCastOp::create(b, loc, i64, ivIndex);
	mlir::Value ivF64 = mlir::arith::SIToFPOp::create(b, loc, f64, ivI64);
	mlir::Value denomF64 = mlir::arith::SIToFPOp::create(b, loc, f64, denomI64);

	mlir::Value span = mlir::arith::SubFOp::create(b, loc, cUpper, cLower);
	mlir::Value ratio = mlir::arith::DivFOp::create(b, loc, ivF64, denomF64);
	mlir::Value offset = mlir::arith::MulFOp::create(b, loc, span, ratio);
	return mlir::arith::AddFOp::create(b, loc, cLower, offset);
}

}

#endif //EZ_COMPILE_LOWER_UTIL_H
