//===-- Linker.cpp ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Linker.h"

#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/FileSystem.h"

#include <array>

namespace ezcompile::link {

//===----------------------------------------------------------------------===//
// 数学函数列表
//===----------------------------------------------------------------------===//

static const char* mathFunctions[] = {
    "pow", "powf", "powl", "sqrt", "sqrtf", "sqrtl",
    "sin", "sinf", "sinl", "cos", "cosf", "cosl",
    "tan", "tanf", "tanl", "exp", "expf", "expl",
    "log", "logf", "logl", "log2", "log2f", "log2l",
    "log10", "log10f", "log10l", "floor", "floorf", "floorl",
    "ceil", "ceilf", "ceill", "round", "roundf", "roundl",
    "fabs", "fabsf", "fabsl", "fmod", "fmodf", "fmodl",
    // LLVM intrinsics
    "llvm.pow.f32", "llvm.pow.f64", "llvm.sqrt.f32", "llvm.sqrt.f64",
    "llvm.sin.f32", "llvm.sin.f64", "llvm.cos.f32", "llvm.cos.f64",
    "llvm.exp.f32", "llvm.exp.f64", "llvm.log.f32", "llvm.log.f64",
    nullptr
};

//===----------------------------------------------------------------------===//
// LibraryDetector
//===----------------------------------------------------------------------===//

bool LibraryDetector::isMathFunction(const std::string &funcName) {
    for (const char** p = mathFunctions; *p; ++p) {
        if (funcName == *p) return true;
    }
    return false;
}

std::string LibraryDetector::getMathLibraryName(const llvm::Triple &triple) {
    switch (triple.getOS()) {
    case llvm::Triple::MacOSX:
    case llvm::Triple::IOS:
    case llvm::Triple::Win32:
        return "";  // 内置
    default:
        return "m"; // Unix-like
    }
}

DetectedLibraries LibraryDetector::detect(llvm::Module &module,
                                           const std::string &targetTripleStr) {
    DetectedLibraries result;
    std::string tripleStr = targetTripleStr.empty() 
        ? module.getTargetTriple().str() : targetTripleStr;
    llvm::Triple triple(tripleStr);
    
    bool needMath = false;
    for (const llvm::Function &func : module.functions()) {
        if (func.isDeclaration()) {
            std::string name = func.getName().str();
            result.externalFunctions.insert(name);
            if (isMathFunction(name)) needMath = true;
        }
    }
    
    if (needMath) {
        std::string lib = getMathLibraryName(triple);
        if (!lib.empty()) result.libraries.insert(lib);
    }
    
    return result;
}

//===----------------------------------------------------------------------===//
// Linker
//===----------------------------------------------------------------------===//

std::string Linker::findClang() {
    if (auto clang = llvm::sys::findProgramByName("clang"))
        return *clang;
    if (llvm::sys::fs::exists("/usr/bin/clang"))
        return "/usr/bin/clang";
    return "clang";
}

std::vector<std::string> Linker::buildCommandLine() const {
    std::vector<std::string> args;
    args.push_back(findClang());
    
    if (!config.targetTriple.empty()) {
        args.push_back("-target");
        args.push_back(config.targetTriple);
    }
    
    args.push_back("-o");
    args.push_back(config.outputFile);
    
    for (const auto &lib : config.libraries)
        args.push_back("-l" + lib);
    
    args.push_back(config.objectFile);
    
    if (config.verbose) args.push_back("-v");
    
    return args;
}

mlir::LogicalResult Linker::run() {
    auto args = buildCommandLine();
    
    if (config.verbose) {
        llvm::errs() << "Linking:";
        for (const auto &arg : args) llvm::errs() << " " << arg;
        llvm::errs() << "\n";
    }
    
    std::vector<llvm::StringRef> refArgs(args.begin(), args.end());
    std::array<std::optional<llvm::StringRef>, 3> redirects;
    
    int result = llvm::sys::ExecuteAndWait(
        args[0], refArgs, std::nullopt, redirects, 0, 0);
    
    if (result != 0) {
        llvm::errs() << "Error: linker failed with exit code " << result << "\n";
        return mlir::failure();
    }
    
    llvm::errs() << "Link succeeded: " << config.outputFile << "\n";
    return mlir::success();
}

mlir::LogicalResult Linker::linkModule(llvm::Module &module,
                                        const std::string &objectFile,
                                        const std::string &outputFile,
                                        const std::string &targetTriple) {
    auto detected = LibraryDetector::detect(module, targetTriple);
    
    if (!detected.libraries.empty()) {
        llvm::errs() << "Required libraries:";
        for (const auto &lib : detected.libraries)
            llvm::errs() << " -l" << lib;
        llvm::errs() << "\n";
    }
    
    LinkerConfig config;
    config.objectFile = objectFile;
    config.outputFile = outputFile;
    config.targetTriple = targetTriple;
    config.libraries = std::vector<std::string>(detected.libraries.begin(), 
                                                 detected.libraries.end());
    
    return Linker(config).run();
}

} // namespace ezcompile::link