//===-- BuilderUtil.h ------------------------------------------*- C++ -*-===//
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


#ifndef EZ_COMPILE_BUILDER_UTIL_H
#define EZ_COMPILE_BUILDER_UTIL_H

#include "mlir/Dialect/Arith/IR/Arith.h"

namespace ezcompile {

inline mlir::Value constIndex(mlir::OpBuilder &builder, mlir::Location location, int64_t value) {
	return mlir::arith::ConstantIndexOp::create(builder, location, value);
}

// 将任何数值/索引值转换为 f64（用于 yield 值和坐标计算）
inline mlir::Value castToF64(mlir::OpBuilder& b, mlir::Location loc, mlir::Value v) {
	mlir::Type t = v.getType();
	mlir::Type f64 = b.getF64Type();

	if (t == f64) return v;

	if (auto ft = dyn_cast<mlir::FloatType>(t)) {
		if (ft.getWidth() < 64) {
			return mlir::arith::ExtFOp::create(b, loc, f64, v);
		}
		if (ft.getWidth() > 64) {
			return mlir::arith::TruncFOp::create(b, loc, f64, v);
		}
	}

	if (llvm::isa<mlir::IndexType>(t)) {
		auto i64 = b.getI64Type();
		mlir::Value asI64 = mlir::arith::IndexCastOp::create(b, loc, i64, v);
		return mlir::arith::SIToFPOp::create(b, loc, f64, asI64);
	}

	if (auto it = dyn_cast<mlir::IntegerType>(t)) {
		// 整型只存在有符号整型
		return mlir::arith::SIToFPOp::create(b, loc, f64, v);
	}

	// 应该不存在其它类型
	llvm_unreachable("Unsupported type for castToF64");
}

inline mlir::Value modIndex(mlir::OpBuilder &b,
							  mlir::Location loc,
							  mlir::Value ivIndex,
							  int64_t modulus) {
	if (!ivIndex || !ivIndex.getType().isIndex()) {
		mlir::emitError(loc, "modIndex: ivIndex must be index type");
		return {};
	}
	if (modulus == 0) {
		mlir::emitError(loc, "modIndex: modulus must be non-zero");
		return {};
	}

	mlir::Type i64 = b.getI64Type();
	mlir::Type idx = b.getIndexType();

	mlir::Value cModI64 = mlir::arith::ConstantIntOp::create(b, loc, i64, modulus);

	// index -> i64
	mlir::Value ivI64 = mlir::arith::IndexCastOp::create(b, loc, i64, ivIndex);

	// i64 % i64
	mlir::Value rI64 = mlir::arith::RemSIOp::create(b, loc, ivI64, cModI64);

	// i64 -> index
	return mlir::arith::IndexCastOp::create(b, loc, idx, rI64);
}

inline mlir::Value modInt64(mlir::OpBuilder &b,
						  mlir::Location loc,
						  mlir::Value ivInt,
						  int64_t modulus) {
	if (!ivInt) return {};
	if (modulus == 0) {
		mlir::emitError(loc, "modInt64: modulus must be non-zero");
		return {};
	}

	// 明确拒绝 index：只允许真正的 IntegerType
	if (ivInt.getType().isIndex()) {
		mlir::emitError(loc, "modInt: index type is not allowed; expected integer type");
		return {};
	}

	auto intTy = llvm::dyn_cast<mlir::IntegerType>(ivInt.getType());
	if (!intTy) {
		mlir::emitError(loc, "modInt: ivInt must be an integer type");
		return {};
	}

	// modulus 常量使用同位宽类型（必要时会截断到该位宽）
	mlir::Value cMod = mlir::arith::ConstantIntOp::create(b, loc, intTy, modulus);

	// 整型同位宽取模，结果类型与 ivInt 相同
	return mlir::arith::RemSIOp::create(b, loc, ivInt, cMod);
}

}

#endif //EZ_COMPILE_BUILDER_UTIL_H
