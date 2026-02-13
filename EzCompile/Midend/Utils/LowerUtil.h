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

#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "BuilderUtil.h"

namespace ezcompile {

/// --------- 穿透 unrealized cast 拿到对应的 alloc ---------
inline mlir::Value stripCasts(mlir::Value v) {
	while (auto cast = v.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
		if (cast.getNumOperands() == 0) break;
		v = cast.getOperand(0);
	}
	return v;
}

inline mlir::memref::AllocOp getDefiningFieldOp(mlir::Value maybeFieldLike) {
	mlir::Value base = stripCasts(maybeFieldLike);
	return base.getDefiningOp<mlir::memref::AllocOp>();
}

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

inline mlir::Value lowerSample(mlir::OpBuilder& b, mlir::Location loc,
                               comp::SampleOp sampleOp,
                               mlir::ValueRange spatialIndices) {
    mlir::memref::AllocOp allocOp = getDefiningFieldOp(sampleOp.getField());
    if (!allocOp) {
        sampleOp.emitError() << "cannot resolve defining alloc op for field " << sampleOp.getField();
        return {};
    }
    mlir::Value memref = allocOp.getResult();

    llvm::SmallVector<mlir::Value, 4> accessIndices;

    llvm::ArrayRef<int64_t> shifts = sampleOp.getShift();

    // 校验：运行时提供的空间索引数量必须与静态定义的 shift 数量一致
    if (spatialIndices.size() != shifts.size()) {
        sampleOp.emitError() << "spatial indices count (" << spatialIndices.size()
                             << ") does not match shift count (" << shifts.size() << ")";
        return {};
    }

    // 遍历空间索引并应用偏移
    for (auto it : llvm::zip(spatialIndices, shifts)) {
        mlir::Value iv = std::get<0>(it);
        int64_t shiftVal = std::get<1>(it);
        if (shiftVal == 0) {
            accessIndices.push_back(iv);
        } else {
            mlir::Value cShift = b.create<mlir::arith::ConstantIndexOp>(loc, shiftVal);
            mlir::Value shiftedIv = b.create<mlir::arith::AddIOp>(loc, iv, cShift);
            accessIndices.push_back(shiftedIv);
        }
    }

    return b.create<mlir::memref::LoadOp>(loc, memref, accessIndices);
}

}

#endif //EZ_COMPILE_LOWER_UTIL_H
