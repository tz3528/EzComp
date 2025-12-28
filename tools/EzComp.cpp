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

#include "mlir/IR/AsmState.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "EzCompile/Frontend/include/Parser.h"
#include "EzCompile/Frontend/include/AST.h"
#include "EzCompile/Frontend/include/Semantic.h"

namespace cl = llvm::cl;

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

/// Returns a Toy AST resulting from parsing the file or a nullptr on error.
static std::unique_ptr<ezcompile::ParsedModule> parseInputFile(llvm::StringRef filename) {
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
        llvm::MemoryBuffer::getFileOrSTDIN(filename);
    if (std::error_code ec = fileOrErr.getError()) {
        llvm::errs() << "Could not open input file: " << ec.message() << "\n";
        return nullptr;
    }

    auto out = std::make_unique<ezcompile::ParsedModule>();

    out->bufferID = static_cast<int>(
        out->sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc())
    );

    ezcompile::Lexer lexer(out->sourceMgr, out->bufferID, &out->context);
    ezcompile::Parser parser(lexer, out->sourceMgr, out->bufferID, &out->context);

    out->module = parser.parseModule();
    if (!out->module) return nullptr;

    if (parser.hadError()) return nullptr;

    ezcompile::Semantic semantic(out->sourceMgr,out->bufferID,&out->context);

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

    ezcompile::dump(*moduleAST->module);
    return 0;
}

int main(int argc, char **argv) {
    mlir::DialectRegistry registry;

    // Register any command line options.
    mlir::registerAsmPrinterCLOptions();
    mlir::registerMLIRContextCLOptions();
    cl::ParseCommandLineOptions(argc, argv, "comp compiler\n");

    switch (emitAction) {
    case Action::DumpAST:
        return dumpAST();
    default:
        llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
    }

    return mlir::asMainReturnCode(mlir::MlirOptMain(
        argc, argv, "Minimal Standalone optimizer driver\n", registry));
}