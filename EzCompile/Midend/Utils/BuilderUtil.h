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

}

#endif //EZ_COMPILE_BUILDER_UTIL_H
