//===-- LowerCompField.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// 将 comp.field 操作降级为 memref
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

struct FieldOpLowering : mlir::OpConversionPattern<comp::FieldOp> {
	using OpConversionPattern<comp::FieldOp>::OpConversionPattern;

	mlir::LogicalResult matchAndRewrite(comp::FieldOp fieldOp, OpAdaptor adaptor,
								  mlir::ConversionPatternRewriter &rewriter) const override {
		// 1. 获取 Field 的基础信息
		auto spaceDims = fieldOp.getSpaceDims();
		auto fieldType = llvm::cast<comp::FieldType>(fieldOp.getResult().getType());
		mlir::Type elementType = fieldType.getElementType();

		// 2. 构建 MemRef 的 Shape
		mlir::SmallVector<int64_t, 4> shape;
		shape.push_back(2);

		// 3. 解析空间维度大小
		for (auto dimAttr : spaceDims) {
			auto dimRef = llvm::cast<mlir::FlatSymbolRefAttr>(dimAttr);
			auto dimOp = mlir::SymbolTable::lookupNearestSymbolFrom<comp::DimOp>(fieldOp, dimRef);

			if (!dimOp) {
				return fieldOp.emitError("referenced dimension '")
					   << dimRef.getValue() << "' not found.";
			}

			shape.push_back(dimOp.getPoints());
		}

		// 4. 创建 MemRef 类型
		auto memRefType = mlir::MemRefType::get(shape, elementType);

		// 5. 生成 AllocOp
		mlir::Location loc = fieldOp.getLoc();
		auto alloc = mlir::memref::AllocOp::create(rewriter, loc, memRefType);

		// 6. 替换原 Op
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
		// 许所有未明确指定的类型保持原样（Legal）
		typeConverter.addConversion([](mlir::Type type) { return type; });
		// 指定 FieldType 应该变为什么，用于类型检查和 Materialization 触发
		typeConverter.addConversion([](comp::FieldType type) -> mlir::Type {
			// 转换目标可以是具体的 Ranked MemRef，也可以是 Unranked，这里主要用于信号握手
			return mlir::UnrankedMemRefType::get(type.getElementType(), 0);
		});
		// 当框架发现 Op 被替换成了 MemRef，但下游 Op 还在索要 !comp.field 时，会回调这里
		typeConverter.addSourceMaterialization([&](mlir::OpBuilder &builder, mlir::Type type,
											   mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
			return mlir::UnrealizedConversionCastOp::create(builder, loc, type, inputs).getResult(0);
		});
		// 配置 Target Materialization
		typeConverter.addTargetMaterialization([&](mlir::OpBuilder &builder, mlir::Type type,
												   mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
			return mlir::UnrealizedConversionCastOp::create(builder, loc, type, inputs).getResult(0);
		});

		// 设置转换目标
		mlir::ConversionTarget target(*context);

		// MemRef 是合法的，Comp::FieldOp 是非法的
		target.addLegalDialect<mlir::memref::MemRefDialect>();
		target.addIllegalOp<comp::FieldOp>();

		// 标记其他 comp ops 为合法
		target.addLegalDialect<comp::CompDialect>();

		mlir::RewritePatternSet patterns(context);
		patterns.add<FieldOpLowering>(typeConverter, context);

		// 应用部分转换
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

}