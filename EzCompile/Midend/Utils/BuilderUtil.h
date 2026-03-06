//===-- BuilderUtil.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Builder 工具函数
// 提供 MLIR OpBuilder 的常用辅助函数
//
//===----------------------------------------------------------------------===//


#ifndef EZ_COMPILE_BUILDER_UTIL_H
#define EZ_COMPILE_BUILDER_UTIL_H

#include "mlir/Dialect/Arith/IR/Arith.h"

namespace ezcompile {

/// 创建常量索引值
inline mlir::Value constIndex(mlir::OpBuilder &builder, mlir::Location location, int64_t value) {
	return mlir::arith::ConstantIndexOp::create(builder, location, value);
}

/// 将任意数值类型转换为 i64（支持 float/index/integer）
inline mlir::Value castToI64(mlir::OpBuilder& b, mlir::Location loc, mlir::Value v) {
	mlir::Type t = v.getType();
	mlir::Type i64 = b.getI64Type();

	if (t == i64) return v;

	if (auto it = dyn_cast<mlir::IntegerType>(t)) {
		if (it.getWidth() < 64) {
			return mlir::arith::ExtSIOp::create(b, loc, i64, v);
		}
		if (it.getWidth() > 64) {
			return mlir::arith::TruncIOp::create(b, loc, i64, v);
		}
	}

	if (llvm::isa<mlir::IndexType>(t)) {
		return mlir::arith::IndexCastOp::create(b, loc, i64, v);
	}

	if (llvm::isa<mlir::FloatType>(t)) {
		return mlir::arith::FPToSIOp::create(b, loc, i64, v);
	}

	llvm_unreachable("Unsupported type for castToI64");
}

/// 将任意数值类型转换为 f64（支持 float/index/integer）
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
		return mlir::arith::SIToFPOp::create(b, loc, f64, v);
	}

	llvm_unreachable("Unsupported type for castToF64");
}

/// 索引类型取模：index -> i64 -> rem -> index
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
	mlir::Value ivI64 = mlir::arith::IndexCastOp::create(b, loc, i64, ivIndex);
	mlir::Value rI64 = mlir::arith::RemSIOp::create(b, loc, ivI64, cModI64);

	return mlir::arith::IndexCastOp::create(b, loc, idx, rI64);
}

/// 整数类型取模（不支持 index 类型）
inline mlir::Value modInt64(mlir::OpBuilder &b,
						  mlir::Location loc,
						  mlir::Value ivInt,
						  int64_t modulus) {
	if (!ivInt) return {};
	if (modulus == 0) {
		mlir::emitError(loc, "modInt64: modulus must be non-zero");
		return {};
	}

	if (ivInt.getType().isIndex()) {
		mlir::emitError(loc, "modInt: index type is not allowed; expected integer type");
		return {};
	}

	auto intTy = llvm::dyn_cast<mlir::IntegerType>(ivInt.getType());
	if (!intTy) {
		mlir::emitError(loc, "modInt: ivInt must be an integer type");
		return {};
	}

	mlir::Value cMod = mlir::arith::ConstantIntOp::create(b, loc, intTy, modulus);
	return mlir::arith::RemSIOp::create(b, loc, ivInt, cMod);
}

} // namespace ezcompile

#endif //EZ_COMPILE_BUILDER_UTIL_H