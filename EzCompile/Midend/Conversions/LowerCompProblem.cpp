//===-- LowerCompProblem.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// comp.problem 降级实现
// 将顶层 comp.problem 操作降级为 func.main 函数
//
//===----------------------------------------------------------------------===//


#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Dialects/Comp/include/Comp.h"

namespace ezcompile {

// 创建/复用 timer_start 声明：() -> void
static mlir::func::FuncOp getOrCreateTimerStartDecl(mlir::ModuleOp module,
													 mlir::OpBuilder &b,
													 mlir::Location loc) {
	mlir::MLIRContext *ctx = module.getContext();
	std::string fnName = "timer_start";

	if (auto existing = module.lookupSymbol<mlir::func::FuncOp>(fnName))
		return existing;

	auto fnTy = mlir::FunctionType::get(ctx, {}, {});

	mlir::OpBuilder::InsertionGuard g(b);
	b.setInsertionPointToStart(module.getBody());
	auto f = b.create<mlir::func::FuncOp>(loc, fnName, fnTy);
	f.setPrivate();
	return f;
}

// 创建/复用 timer_stop_and_print 声明：() -> void
static mlir::func::FuncOp getOrCreateTimerStopDecl(mlir::ModuleOp module,
													mlir::OpBuilder &b,
													mlir::Location loc) {
	mlir::MLIRContext *ctx = module.getContext();
	std::string fnName = "timer_stop_and_print";

	if (auto existing = module.lookupSymbol<mlir::func::FuncOp>(fnName))
		return existing;

	auto fnTy = mlir::FunctionType::get(ctx, {}, {});

	mlir::OpBuilder::InsertionGuard g(b);
	b.setInsertionPointToStart(module.getBody());
	auto f = b.create<mlir::func::FuncOp>(loc, fnName, fnTy);
	f.setPrivate();
	return f;
}

static mlir::func::FuncOp getOrCreateDumpDecl(mlir::ModuleOp module,
											  mlir::OpBuilder &b,
											  mlir::Location loc,
											  mlir::MemRefType memrefTy) {
	mlir::MLIRContext *ctx = module.getContext();
	int64_t rank = memrefTy.getRank();

	std::string fnName = "dump_result_hdf5_f64_rank" + std::to_string(rank);
	if (auto existing = module.lookupSymbol<mlir::func::FuncOp>(fnName))
		return existing;

	// 使用 builtin i32/i64 + LLVM dialect ptr 类型(!llvm.ptr)
	auto i64Ty = b.getI64Type();
	auto i32Ty = b.getI32Type();
	auto llvmPtrTy = mlir::LLVM::LLVMPointerType::get(ctx); // !llvm.ptr :contentReference[oaicite:2]{index=2}

	// (memref<...xf64>, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
	auto fnTy = mlir::FunctionType::get(ctx,
										{memrefTy, i64Ty, llvmPtrTy, llvmPtrTy, llvmPtrTy},
										{i32Ty});

	mlir::OpBuilder::InsertionGuard g(b);
	b.setInsertionPointToStart(module.getBody());
	auto f = b.create<mlir::func::FuncOp>(loc, fnName, fnTy);
	f.setPrivate();
	return f;
}

// 创建/复用一个 LLVM string global（llvm.mlir.global），并返回其首字符地址 !llvm.ptr
static mlir::Value getOrCreateCStringPtr(mlir::ModuleOp module,
										 mlir::OpBuilder &b,
										 mlir::Location loc,
										 llvm::StringRef s) {
	mlir::MLIRContext *ctx = module.getContext();
	std::string sym = ("__comp_dimname_" + s).str();

	// 注意：LLVM dialect string global 不会自动补 '\0' :contentReference[oaicite:3]{index=3}
	std::string nul = (s + "\0").str();
	int64_t len = (int64_t)nul.size();

	auto i8Ty = mlir::IntegerType::get(ctx, 8);
	auto arrTy = mlir::LLVM::LLVMArrayType::get(i8Ty, len); // !llvm.array<len x i8>

	auto glob = module.lookupSymbol<mlir::LLVM::GlobalOp>(sym);
	if (!glob) {
		mlir::OpBuilder::InsertionGuard g(b);
		b.setInsertionPointToStart(module.getBody());
		// llvm.mlir.global private constant @sym("...") : !llvm.array<...>
		glob = b.create<mlir::LLVM::GlobalOp>(
				loc, arrTy, true, mlir::LLVM::Linkage::Internal,
				sym, b.getStringAttr(nul));
	}

	// %addr = llvm.mlir.addressof @sym : !llvm.ptr
	auto addr = b.create<mlir::LLVM::AddressOfOp>(loc, glob);

	// GEP [0,0] 取首字符地址；GEPOp 需要 elem_type :contentReference[oaicite:4]{index=4}
	auto llvmPtrTy = mlir::LLVM::LLVMPointerType::get(ctx);
	auto gep = b.create<mlir::LLVM::GEPOp>(
			loc, llvmPtrTy, arrTy, addr,
			llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 0},
			mlir::LLVM::GEPNoWrapFlags::inbounds);

	return gep.getResult();
}

/// 降级 Pattern：将 comp.problem 转换为 func.main
///
/// 实现思路：
/// 1. 创建空的 main 函数（无参数、无返回值）
/// 2. 将 problem body 内的所有操作移动到 main 函数中
/// 3. 删除原 problem 操作
struct LowerProblemPattern : mlir::OpConversionPattern<comp::ProblemOp> {
	using OpConversionPattern<comp::ProblemOp>::OpConversionPattern;

	mlir::LogicalResult matchAndRewrite(comp::ProblemOp op,
								  OpAdaptor adaptor,
								  mlir::ConversionPatternRewriter& rewriter) const override {
		mlir::Region &body = op.getBody();
		if (!llvm::hasSingleElement(body)) {
			return rewriter.notifyMatchFailure(
					op, "comp.problem body has multiple blocks; cannot inline safely");
		}
		mlir::Block &problemBlock = body.front();

		// 扫 dim：找到 timeDim（DimOp::getTimeVar()==true），其余作为 spaceDims（声明顺序）
		llvm::SmallVector<comp::DimOp, 8> spaceDims;
		comp::DimOp timeDim;
		for (auto d : problemBlock.getOps<comp::DimOp>()) {
			if (d.getTimeVar()) {
				timeDim = d;
			}
			else {
				spaceDims.emplace_back(d);
			}
		}
		if (!timeDim)
			return rewriter.notifyMatchFailure(op, "no time dim found (DimOp::getTimeVar()==true)");

		// 创建 main
		rewriter.setInsertionPoint(op);
		auto funcType = rewriter.getFunctionType(/*inputs=*/{}, /*results=*/{});
		auto mainFunc = rewriter.create<mlir::func::FuncOp>(op.getLoc(), "main", funcType);
		mlir::Block *entry = mainFunc.addEntryBlock();

		rewriter.setInsertionPointToEnd(entry);

		// ===== 位置1：开始计时 =====
		auto module = op->getParentOfType<mlir::ModuleOp>();
		auto timerStartDecl = getOrCreateTimerStartDecl(module, rewriter, op.getLoc());
		rewriter.create<mlir::func::CallOp>(op.getLoc(), timerStartDecl, mlir::ValueRange());

		llvm::SmallVector<mlir::Operation *, 16> opsToMove;
		opsToMove.reserve(problemBlock.getOperations().size());
		for (mlir::Operation &inner : problemBlock.getOperations()) {
			opsToMove.push_back(&inner);
		}

		for (mlir::Operation *inner : opsToMove) {
			if (inner->hasTrait<mlir::OpTrait::IsTerminator>()) {
				continue;
			}
			if (isa<comp::DimOp>(inner)) {
				continue;
			}
			rewriter.moveOpBefore(inner, entry, entry->end());
		}

		// 找唯一 memref.alloc
		mlir::memref::AllocOp alloc;
		for (auto &opInMain : entry->getOperations()) {
			if (auto a = dyn_cast<mlir::memref::AllocOp>(opInMain)) {
				if (alloc) {
					return rewriter.notifyMatchFailure(op, "more than one memref.alloc found");
				}
				alloc = a;
			}
		}
		if (!alloc)
			return rewriter.notifyMatchFailure(op, "no memref.alloc found (expected exactly one)");

		auto memrefTy = dyn_cast<mlir::MemRefType>(alloc.getResult().getType());
		if (!memrefTy)
			return rewriter.notifyMatchFailure(op, "alloc result is not a memref type");

		int64_t rank = memrefTy.getRank();
		if (rank < 2 || rank > 4)
			return rewriter.notifyMatchFailure(op, "dump only supports rank 2..4");

		// ===== 位置2：停止计时 =====
		auto timerStopDecl = getOrCreateTimerStopDecl(module, rewriter, op.getLoc());
		rewriter.create<mlir::func::CallOp>(op.getLoc(), timerStopDecl, mlir::ValueRange());

		// timeIndex = (tPoints % 2) as i64 constant
		int64_t timeIndex = (int64_t)timeDim.getPoints() - 1;
		auto timeIndexConst = rewriter.create<mlir::arith::ConstantOp>(
				op.getLoc(), rewriter.getI64IntegerAttr(timeIndex));

		mlir::MLIRContext *ctx = rewriter.getContext();
		auto llvmPtrTy = mlir::LLVM::LLVMPointerType::get(ctx);
		auto f64Ty = rewriter.getF64Type();

		// 为 dimNames/lowers/uppers 在栈上建数组：llvm.alloca (elem_type + arraySize) :contentReference[oaicite:6]{index=6}
		auto nConst = rewriter.create<mlir::LLVM::ConstantOp>(
				op.getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(rank - 1));

		auto dimNamesAlloca = rewriter.create<mlir::LLVM::AllocaOp>(
				op.getLoc(), llvmPtrTy, nConst.getResult(),
				/*alignment=*/rewriter.getI64IntegerAttr(8),
				/*elem_type=*/mlir::TypeAttr::get(llvmPtrTy));

		auto lowersAlloca = rewriter.create<mlir::LLVM::AllocaOp>(
				op.getLoc(), llvmPtrTy, nConst.getResult(),
				/*alignment=*/rewriter.getI64IntegerAttr(8),
				/*elem_type=*/mlir::TypeAttr::get(f64Ty));

		auto uppersAlloca = rewriter.create<mlir::LLVM::AllocaOp>(
				op.getLoc(), llvmPtrTy, nConst.getResult(),
				/*alignment=*/rewriter.getI64IntegerAttr(8),
				/*elem_type=*/mlir::TypeAttr::get(f64Ty));

		for (int64_t i = 0; i < rank - 1; ++i) {
			auto d = spaceDims[i];

			// dimNames[i] = &"name\0"[0]
			mlir::Value cstrPtr = getOrCreateCStringPtr(op->getParentOfType<mlir::ModuleOp>(),
														rewriter, op.getLoc(), d.getSymName());

			// slot ptr = gep base[idx] , elem_type = !llvm.ptr
			auto nameSlot = rewriter.create<mlir::LLVM::GEPOp>(
					op.getLoc(), llvmPtrTy,        // result type: !llvm.ptr
					llvmPtrTy,                           // element type of the pointee array: !llvm.ptr
					dimNamesAlloca.getResult(),         	    // base pointer
					llvm::ArrayRef<mlir::LLVM::GEPArg>{(int32_t)i},
					mlir::LLVM::GEPNoWrapFlags::inbounds);
			rewriter.create<mlir::LLVM::StoreOp>(op.getLoc(), cstrPtr, nameSlot);

			// lowers/uppers：double 常量
			double lo = d.getLower().convertToDouble();
			double up = d.getUpper().convertToDouble();

			auto loC = rewriter.create<mlir::LLVM::ConstantOp>(
					op.getLoc(), f64Ty, rewriter.getF64FloatAttr(lo));
			auto upC = rewriter.create<mlir::LLVM::ConstantOp>(
					op.getLoc(), f64Ty, rewriter.getF64FloatAttr(up));

			auto loSlot = rewriter.create<mlir::LLVM::GEPOp>(
					op.getLoc(), llvmPtrTy, 			 	// result type: !llvm.ptr
					f64Ty,                                    	// element type
					lowersAlloca.getResult(),                        // base pointer: !llvm.ptr
					llvm::ArrayRef<mlir::LLVM::GEPArg>{(int32_t)i},  // constant index
					mlir::LLVM::GEPNoWrapFlags::inbounds);
			auto upSlot = rewriter.create<mlir::LLVM::GEPOp>(
					op.getLoc(), llvmPtrTy,                // result type: !llvm.ptr
					f64Ty,                                          // element type of pointee array: f64
					uppersAlloca.getResult(),                       // base pointer
					llvm::ArrayRef<mlir::LLVM::GEPArg>{(int32_t)i}, // index (Value or int)
					mlir::LLVM::GEPNoWrapFlags::inbounds);

			// 注意：store 的 value type 是 f64，ptr 是 !llvm.ptr（opaque），由 elem_type 约束
			rewriter.create<mlir::LLVM::StoreOp>(op.getLoc(), loC, loSlot);
			rewriter.create<mlir::LLVM::StoreOp>(op.getLoc(), upC, upSlot);
		}

		auto dumpDecl = getOrCreateDumpDecl(module, rewriter, op.getLoc(), memrefTy);

		// func.call @dump_result_hdf5_f64_rank{rank}(memref, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr) : i32
		rewriter.create<mlir::func::CallOp>(
				op.getLoc(), dumpDecl,
				mlir::ValueRange{alloc.getResult(),
								 timeIndexConst.getResult(),
								 dimNamesAlloca.getResult(),
								 lowersAlloca.getResult(),
								 uppersAlloca.getResult()});

		rewriter.create<mlir::func::ReturnOp>(op.getLoc());

		rewriter.eraseOp(op);
		return mlir::success();
	}
};

struct LowerCompProblemPass : mlir::PassWrapper<LowerCompProblemPass, mlir::OperationPass<mlir::ModuleOp>> {
	MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerCompProblemPass)

	void getDependentDialects(mlir::DialectRegistry& registry) const override {
		registry.insert<
			mlir::affine::AffineDialect, mlir::memref::MemRefDialect,
			mlir::arith::ArithDialect, mlir::func::FuncDialect, mlir::LLVM::LLVMDialect
		>();
	}

	mlir::StringRef getArgument() const override { return "lower-comp-problem"; }

	void runOnOperation() override {
		mlir::MLIRContext* context = &getContext();
		mlir::ModuleOp module = getOperation();

		mlir::ConversionTarget target(*context);

		target.addLegalDialect<
			mlir::affine::AffineDialect, mlir::memref::MemRefDialect,
			mlir::arith::ArithDialect, mlir::func::FuncDialect, mlir::LLVM::LLVMDialect>();
		target.addIllegalOp<comp::ProblemOp>();

		mlir::RewritePatternSet patterns(context);
		patterns.add<LowerProblemPattern>(context);

		if (mlir::failed(applyPartialConversion(module, target, std::move(patterns)))) {
			signalPassFailure();
		}
	}
};

void registerLowerCompProblemPass() {
	mlir::PassRegistration<LowerCompProblemPass>();
}

std::unique_ptr<mlir::Pass> createLowerCompProblemPass() {
	return std::make_unique<LowerCompProblemPass>();
}

} // namespace ezcompile