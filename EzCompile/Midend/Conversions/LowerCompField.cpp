//===-- LowerCompField.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// comp.field 降级实现
// 将字段声明操作降级为 memref.alloc
//
//===----------------------------------------------------------------------===//


#include "llvm/Support/Casting.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Dialects/Comp/include/Comp.h"

namespace ezcompile {

/// 降级 Pattern：将 comp.field 转换为 memref.alloc
///
/// 实现思路：
/// 1. 从 field 的空间维度属性解析各维度的 points 数量
/// 2. 构建 memref shape：第一维为时间缓冲（大小为 2），后续为空间维度
/// 3. 创建 memref.alloc 并替换原操作
struct FieldOpLowering : mlir::OpConversionPattern<comp::FieldOp> {
	using OpConversionPattern<comp::FieldOp>::OpConversionPattern;

	mlir::LogicalResult matchAndRewrite(comp::FieldOp fieldOp, OpAdaptor adaptor,
								  mlir::ConversionPatternRewriter &rewriter) const override {
		auto spaceDims = fieldOp.getSpaceDims();
		auto fieldType = llvm::cast<comp::FieldType>(fieldOp.getResult().getType());
		mlir::Type elementType = fieldType.getElementType();

		// 构建 MemRef shape: [2, space_dim1, space_dim2, ...]
		// 第一维为时间 ping-pong 缓冲
		mlir::SmallVector<int64_t, 4> shape;
		shape.push_back(2);

		for (auto dimAttr : spaceDims) {
			auto dimRef = llvm::cast<mlir::FlatSymbolRefAttr>(dimAttr);
			auto dimOp = mlir::SymbolTable::lookupNearestSymbolFrom<comp::DimOp>(fieldOp, dimRef);

			if (!dimOp) {
				return fieldOp.emitError("referenced dimension '")
					   << dimRef.getValue() << "' not found.";
			}

			shape.push_back(dimOp.getPoints());
		}

		auto memRefType = mlir::MemRefType::get(shape, elementType);

		mlir::Location loc = fieldOp.getLoc();
		auto alloc = mlir::memref::AllocOp::create(rewriter, loc, memRefType);

		rewriter.replaceOp(fieldOp, alloc);

		return mlir::success();
	}
};

struct LowerCompFieldPass : mlir::PassWrapper<LowerCompFieldPass, mlir::OperationPass<mlir::ModuleOp>> {

	MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerCompFieldPass)

	void getDependentDialects(mlir::DialectRegistry &registry) const override {
		registry.insert<mlir::memref::MemRefDialect, comp::CompDialect>();
	}

	mlir::StringRef getArgument() const override { return "lower-comp-field"; }
	mlir::StringRef getDescription() const override {
		return "Lower comp.field to memref.alloc with ping-pong buffering";
	}

	void runOnOperation() override {
		mlir::ModuleOp module = getOperation();
		mlir::MLIRContext *context = &getContext();

		mlir::TypeConverter typeConverter;
		typeConverter.addConversion([](mlir::Type type) { return type; });
		typeConverter.addConversion([](comp::FieldType type) -> mlir::Type {
			return mlir::UnrankedMemRefType::get(type.getElementType(), 0);
		});
		typeConverter.addSourceMaterialization([&](mlir::OpBuilder &builder, mlir::Type type,
											   mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
			return mlir::UnrealizedConversionCastOp::create(builder, loc, type, inputs).getResult(0);
		});
		typeConverter.addTargetMaterialization([&](mlir::OpBuilder &builder, mlir::Type type,
												   mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
			return mlir::UnrealizedConversionCastOp::create(builder, loc, type, inputs).getResult(0);
		});

		mlir::ConversionTarget target(*context);

		target.addLegalDialect<mlir::memref::MemRefDialect>();
		target.addIllegalOp<comp::FieldOp>();
		target.addLegalDialect<comp::CompDialect>();

		mlir::RewritePatternSet patterns(context);
		patterns.add<FieldOpLowering>(typeConverter, context);

		if (mlir::failed(applyPartialConversion(module, target, std::move(patterns)))) {
			signalPassFailure();
		}
	}
};

void registerLowerCompFieldPass() {
	mlir::PassRegistration<LowerCompFieldPass>();
}

std::unique_ptr<mlir::Pass> createLowerCompFieldPass() {
	return std::make_unique<LowerCompFieldPass>();
}

} // namespace ezcompile