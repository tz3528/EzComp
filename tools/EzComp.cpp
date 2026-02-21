//===-- EzComp.cpp ---------------------------------------------*- C++ -*-===//
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

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "EzCompile/Frontend/include/Parser.h"
#include "EzCompile/Frontend/include/AST.h"
#include "EzCompile/Frontend/include/Semantic/Semantic.h"
#include "IRGen/MLIRGen.h"
#include "Transforms/Pipelines.h"
#include "Transforms/Passes.h"

namespace cl = llvm::cl;
using namespace ezcompile;

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
enum Action { None, DumpAST, DumpMLIR };
} // namespace
static cl::opt<enum Action> emitAction(
    "emit", cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
    cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")));

static mlir::PassPipelineCLParser passPipeline("",
    "Run an MLIR pass pipeline (use --pass-pipeline=...)");

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
    if (!moduleAST)
        return 1;

    auto parse_module = moduleAST.get();

    mlir::DialectRegistry registry;
    mlir::registerAllExtensions(registry);

    mlir::MLIRContext context(registry);

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

int main(int argc, char **argv) {
    mlir::DialectRegistry registry;

    // Register any command line options.
    mlir::registerAsmPrinterCLOptions();
    mlir::registerMLIRContextCLOptions();
    registerPasses();
    registerPipelines();
    cl::ParseCommandLineOptions(argc, argv, "comp compiler\n");

    switch (emitAction) {
    case Action::DumpAST:
        return dumpAST();
    case Action::DumpMLIR:
        return dumpMLIR();
    default:
        llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
    }

    return mlir::asMainReturnCode(mlir::MlirOptMain(
        argc, argv, "Minimal Standalone optimizer driver\n", registry));
}