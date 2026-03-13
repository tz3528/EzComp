//===-- EzComp.cpp ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// EzComp 编译器主入口
// 提供 comp 语言编译器的命令行驱动，包括：
// - 源文件解析
// - AST 和 MLIR 输出
// - 降级管线配置
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/AsmState.h"

#include "EzCompile/Frontend/include/Parser.h"
#include "EzCompile/Frontend/include/AST.h"
#include "EzCompile/Frontend/include/Semantic/Semantic.h"
#include "IRGen/MLIRGen.h"
#include "Transforms/LowerPipelines.h"
#include "Transforms/LowerPasses.h"
#include "Transforms/OptPasses.h"
#include "Transforms/OptPipelines.h"
#include "Driver/BackendDriver.h"
#include "Driver/BackendOptions.h"

namespace cl = llvm::cl;
using namespace ezcompile;
using namespace ezresearch;

/// 仅注册项目需要的 LLVM 转换扩展，避免链接不需要的库
static void registerNeededExtensions(mlir::DialectRegistry &registry) {
    // LLVM 转换扩展
    mlir::arith::registerConvertArithToLLVMInterface(registry);
    mlir::registerConvertComplexToLLVMInterface(registry);
    mlir::cf::registerConvertControlFlowToLLVMInterface(registry);
    mlir::registerConvertFuncToLLVMInterface(registry);
    mlir::registerConvertMathToLLVMInterface(registry);
    mlir::registerConvertMemRefToLLVMInterface(registry);
    mlir::ub::registerConvertUBToLLVMInterface(registry);
    mlir::vector::registerConvertVectorToLLVMInterface(registry);
    // Func 内联扩展
    mlir::func::registerInlinerExtension(registry);
}

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input toy file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

namespace {
enum InputType { Comp, MLIR };
} // namespace
static cl::opt<enum InputType> inputType(
    "x", cl::init(Comp), cl::desc("Decided the kind of output desired"),
    cl::values(clEnumValN(Comp, "comp", "load the input file as a Toy source.")),
    cl::values(clEnumValN(MLIR, "mlir",
                          "load the input file as an MLIR file")));

namespace {
enum Action { None, DumpAST, DumpMLIR , DumpLLVMIR , Compile };
} // namespace
static cl::opt<enum Action> emitAction(
    "emit", cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
    cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")),
    cl::values(clEnumValN(DumpLLVMIR, "llvmir", "output the LLVM dump")),
    cl::values(clEnumValN(Compile, "compile", "compile to executable")));

static mlir::PassPipelineCLParser passPipeline("",
    "Run an MLIR pass pipeline (use --pass-pipeline=...)");

// 全局变量：记录是否启用了向量化（用于决定是否启用 AVX）
static bool useVectorize = false;

/// Returns a Toy AST resulting from parsing the file or a nullptr on error.
static std::unique_ptr<ParsedModule> parseInputFile(llvm::StringRef filename) {
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
        llvm::MemoryBuffer::getFileOrSTDIN(filename);
    if (std::error_code ec = fileOrErr.getError()) {
        llvm::errs() << "Could not open input file: " << ec.message() << "\n";
        return nullptr;
    }

    auto out = std::make_unique<ParsedModule>();

    out->bufferID = static_cast<int>(
        out->sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc())
    );

    auto ctx = mlir::MLIRContext();

    Lexer lexer(out->sourceMgr, out->bufferID, &ctx);
    Parser parser(lexer, out->sourceMgr, out->bufferID, &ctx);

    out->module = parser.parseModule();
    if (!out->module) return nullptr;

    if (parser.hadError()) return nullptr;

    Semantic semantic(out->sourceMgr,out->bufferID,&ctx);

    out->sema = semantic.analyze(*out->module);

    if (!out->sema) return nullptr;

    if (semantic.hadError()) return nullptr;

    return out;
}

static void LoadDialect(mlir::MLIRContext &context) {
    // 加载所有需要的方言
    context.getOrLoadDialect<comp::CompDialect>();
    context.getOrLoadDialect<mlir::affine::AffineDialect>();
    context.getOrLoadDialect<mlir::arith::ArithDialect>();
    context.getOrLoadDialect<mlir::math::MathDialect>();
    context.getOrLoadDialect<mlir::memref::MemRefDialect>();
    context.getOrLoadDialect<mlir::scf::SCFDialect>();
    context.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
    context.getOrLoadDialect<mlir::ub::UBDialect>();
}

struct PipelineOptions : LoweringOptions, OptimizationOptions {};

void buildPipeline(mlir::OpPassManager &pm, const PipelineOptions &opt) {
    // 根据选项设置全局向量化标志（用于后续决定是否启用 AVX）
    useVectorize = opt.enableAffineVevtorize.getValue();

    //===--------------------------------------------------------------------===//
    // 阶段1：Comp → Base
    //===--------------------------------------------------------------------===//
	if (opt.enableLowerToBase.getValue() || opt.enableToLLVM.getValue()) {
	    LowerToBase(pm);
	    if (opt.enableHoistBoundary.getValue()) {
	        HoistBoundary(pm);
	    }

	    if (opt.enableAffineVevtorize.getValue()) {
	        AffineVectorize(pm);
	        LoopPeeling(pm);
	    }
	}

	//===--------------------------------------------------------------------===//
	// 阶段2：Affine → SCF
	//===--------------------------------------------------------------------===//
	if (opt.enableAffineToSCF.getValue() || opt.enableToLLVM.getValue()) {
	    AffineToSCF(pm);
	}

	//===--------------------------------------------------------------------===//
	// 阶段3：SCF → ControlFlow
	//===--------------------------------------------------------------------===//
	if (opt.enableSCFToCF.getValue() || opt.enableToLLVM.getValue()) {
	    SCFToCF(pm);
	}

	//===--------------------------------------------------------------------===//
	// 阶段4：基础方言 → LLVM
	//===--------------------------------------------------------------------===//
	if (opt.enableToLLVM.getValue()) {
	    if (opt.enableAffineVevtorize.getValue()) {
	        pm.addPass(mlir::createConvertVectorToLLVMPass());
	    }
	    ToLLVM(pm);
	}
}

void registerPipelines() {
	mlir::PassPipelineRegistration<PipelineOptions>(
		"lowering",
		"Lower Comp dialect via staged lowering with configurable options",
		buildPipeline);
}

static int dumpAST() {
    if (inputType == InputType::MLIR) {
        llvm::errs() << "Can't dump a Comp AST when the input is MLIR\n";
        return 5;
    }

    auto moduleAST = parseInputFile(inputFilename);
    if (!moduleAST)
        return 1;

    dump(*moduleAST->module);
    return 0;
}

static int dumpMLIR() {
    if (inputType == InputType::MLIR) {
        llvm::errs() << "Can't dump a Comp AST when the input is MLIR\n";
        return 5;
    }

    auto moduleAST = parseInputFile(inputFilename);
    if (!moduleAST) {
        return 1;
    }

    auto parse_module = moduleAST.get();

    mlir::DialectRegistry registry;
    registerNeededExtensions(registry);
    mlir::MLIRContext context(registry);
    LoadDialect(context);

    MLIRGen gen(*parse_module, context);
    auto mo = gen.mlirGen();

    if (mlir::failed(mo)) {
        return 2;
    }

    if (passPipeline.hasAnyOccurrences()) {
        mlir::PassManager pm(&context);

        auto errHandler = [&](const llvm::Twine &msg) -> mlir::LogicalResult {
            llvm::errs() << "Failed to parse/add pass pipeline: " << msg << "\n";
            return mlir::failure();
        };

        if (mlir::failed(passPipeline.addToPipeline(pm, errHandler))) {
            return 3;
        }

        if (mlir::failed(pm.run(*mo))) {
            llvm::errs() << "Pipeline failed\n";
            mo->print(llvm::errs());
            return 3;
        }
    }

    gen.print(llvm::outs());

    return 0;
}

static int dumpLLVMIR() {
    if (inputType == InputType::MLIR) {
        llvm::errs() << "Can't dump a Comp AST when the input is MLIR\n";
        return 5;
    }

    auto moduleAST = parseInputFile(inputFilename);
    if (!moduleAST) {
        return 1;
    }

    mlir::DialectRegistry registry;
    registerNeededExtensions(registry);
    mlir::MLIRContext context(registry);
    LoadDialect(context);

    MLIRGen gen(*moduleAST.get(), context);
    auto mo = gen.mlirGen();

    if (mlir::failed(mo)) {
        return 2;
    }

    mlir::PassManager pm(&context);

    // 如果用户指定了 passPipeline，使用用户的配置；否则使用默认的完整 pipeline
    if (passPipeline.hasAnyOccurrences()) {
        auto errHandler = [&](const llvm::Twine &msg) -> mlir::LogicalResult {
            llvm::errs() << "Failed to parse/add pass pipeline: " << msg << "\n";
            return mlir::failure();
        };

        if (mlir::failed(passPipeline.addToPipeline(pm, errHandler))) {
            return 3;
        }
    } else {
        // 默认开启所有优化选项，完整降级到 LLVM 方言
        PipelineOptions opt;
        opt.enableLowerToBase = true;
        opt.enableAffineToSCF = true;
        opt.enableSCFToCF = true;
        opt.enableToLLVM = true;
        opt.enableHoistBoundary=true;
        opt.enableAffineVevtorize=true;
        buildPipeline(pm, opt);
    }

    if (mlir::failed(pm.run(*mo))) {
        llvm::errs() << "Pipeline failed\n";
        mo->print(llvm::errs());
        return 3;
    }

    // DumpLLVMIR 模式：不考虑选项，直接输出 LLVM IR
    Backend backend(backend::BackendConfig::forDumpLLVMIR());

    if (mlir::failed(backend.run(*mo))) {
        return 4;
    }

    return 0;
}

static int compile() {
    if (inputType == InputType::MLIR) {
        llvm::errs() << "Can't dump a Comp AST when the input is MLIR\n";
        return 5;
    }

    auto moduleAST = parseInputFile(inputFilename);
    if (!moduleAST) {
        return 1;
    }


    mlir::DialectRegistry registry;
    registerNeededExtensions(registry);
    mlir::MLIRContext context(registry);
    LoadDialect(context);

    MLIRGen gen(*moduleAST.get(), context);
    auto mo = gen.mlirGen();

    if (mlir::failed(mo)) {
        return 2;
    }

    mlir::PassManager pm(&context);

    // 重置全局向量化标志，buildPipeline 会根据选项设置它
    useVectorize = false;

    // 如果用户指定了 passPipeline，使用用户的配置；否则使用默认的完整 pipeline
    if (passPipeline.hasAnyOccurrences()) {
        auto errHandler = [&](const llvm::Twine &msg) -> mlir::LogicalResult {
            llvm::errs() << "Failed to parse/add pass pipeline: " << msg << "\n";
            return mlir::failure();
        };

        if (mlir::failed(passPipeline.addToPipeline(pm, errHandler))) {
            return 3;
        }
    } else {
        // 默认开启所有优化选项，完整降级到 LLVM 方言
        PipelineOptions opt;
        opt.enableLowerToBase = true;
        opt.enableAffineToSCF = true;
        opt.enableSCFToCF = true;
        opt.enableToLLVM = true;
        opt.enableHoistBoundary = true;
        opt.enableAffineVevtorize = true;
        buildPipeline(pm, opt);
    }

    if (mlir::failed(pm.run(*mo))) {
        llvm::errs() << "Pipeline failed\n";
        mo->print(llvm::errs());
        return 3;
    }

    // FullCompile 模式：使用命令行选项进行完整编译
    backend::BackendConfig config = backend::BackendConfig::fromCommandLine(inputFilename);

    // 如果启用了向量化且用户未手动指定 CPU 特性，自动启用 AVX2
    if (useVectorize && config.targetFeaturesVal.empty()) {
        config.targetFeaturesVal = "+avx2";
    }

    Backend backend(config);

    if (mlir::failed(backend.run(*mo))) {
        return 4;
    }

    return 0;
}

int main(int argc, char **argv) {
    // Register any command line options.
    mlir::registerAsmPrinterCLOptions();
    mlir::registerMLIRContextCLOptions();
    registerLowerPasses();
    registerOptPasses();
    registerPipelines();
    cl::ParseCommandLineOptions(argc, argv, "comp compiler\n");

    switch (emitAction) {
    case Action::DumpAST:
        return dumpAST();
    case Action::DumpMLIR:
        return dumpMLIR();
    case Action::DumpLLVMIR:
        return dumpLLVMIR();
    case Action::Compile:
        return compile();
    case Action::None:
        llvm::errs() << "No action specified, use --emit=<action>\n";
        return 1;
    }
    return 1;
}