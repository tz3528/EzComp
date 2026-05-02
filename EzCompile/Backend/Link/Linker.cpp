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
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"

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
// ManifestParser
//===----------------------------------------------------------------------===//

mlir::LogicalResult ManifestParser::load(const fs::path& path, ManifestParser& parser) {
    // 检查文件是否存在
    if (!fs::exists(path)) {
        llvm::errs() << "Manifest file not found: " << path.string() << "\n";
        return mlir::failure();
    }
    
    // 读取文件内容
    auto bufferOrError = llvm::MemoryBuffer::getFile(path.string());
    if (auto ec = bufferOrError.getError()) {
        llvm::errs() << "Error reading manifest file: " << ec.message() << "\n";
        return mlir::failure();
    }
    
    // 解析 JSON
    auto jsonOrError = llvm::json::parse(bufferOrError.get()->getBuffer());
    if (auto ec = jsonOrError.takeError()) {
        llvm::errs() << "Error parsing manifest JSON: " << ec << "\n";
        return mlir::failure();
    }
    
    llvm::json::Object* root = jsonOrError->getAsObject();
    if (!root) {
        llvm::errs() << "Manifest JSON root is not an object\n";
        return mlir::failure();
    }
    
    // 解析 version
    if (auto* versionVal = root->get("version")) {
        if (auto version = versionVal->getAsInteger()) {
            parser.version_ = *version;
        }
    }
    
    // 解析 archives
    if (auto* archivesObj = root->getObject("archives")) {
        for (auto& [key, value] : *archivesObj) {
            if (auto strVal = value.getAsString()) {
                parser.archives_[key.str()] = strVal->str();
            }
        }
    }
    
    // 解析 libraries
    if (auto* libsArray = root->getArray("libraries")) {
        for (auto& libVal : *libsArray) {
            if (auto libStr = libVal.getAsString()) {
                parser.libraries_.push_back(libStr->str());
            }
        }
    }
    
    return mlir::success();
}

mlir::LogicalResult ManifestParser::getString(const std::string& key, std::string& value) const {
    // 支持嵌套 key，用 '/' 分隔
    size_t slashPos = key.find('/');
    
    if (slashPos == std::string::npos) {
        // 顶层 key
        // 目前只支持 version 和 archives 两种顶层 key
        if (key == "version") {
            value = std::to_string(version_);
            return mlir::success();
        }
        return mlir::failure();
    }
    
    // 嵌套 key: "archives/xxx"
    std::string topKey = key.substr(0, slashPos);
    std::string subKey = key.substr(slashPos + 1);
    
    if (topKey == "archives") {
        auto it = archives_.find(subKey);
        if (it != archives_.end()) {
            value = it->second;
            return mlir::success();
        }
    }
    
    return mlir::failure();
}

mlir::LogicalResult ManifestParser::getArchivePath(const std::string& key, std::string& archive) const {
    auto it = archives_.find(key);
    if (it == archives_.end()) {
        llvm::errs() << "Archive not found: " << key << "\n";
        return mlir::failure();
    }
    
    fs::path path(it->second);
    if (path.is_absolute()) {
        // 已经是绝对路径，直接使用
        archive = it->second;
    } else {
        // 相对路径：拼接 baseDir + value
        fs::path fullPath = baseDir / it->second;
        archive = fullPath.string();
    }
    return mlir::success();
}

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

DetectedLibraries LibraryDetector::detect(llvm::Module &module) {
    DetectedLibraries result;
    std::string tripleStr = module.getTargetTriple().str();
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

std::vector<std::string> Linker::buildCommandLine() const {
    std::vector<std::string> args;
    args.push_back(EZCOMP_CXX_COMPILER);
    
    args.push_back(config.objectFile);
    
    args.push_back("-o");
    args.push_back(config.outputFile);

    for (const auto &a : config.archives) {
        args.push_back(a);
    }

    // 收集共享库目录用于 rpath（仅 Linux/macOS 需要）
#if defined(__linux__) || defined(__APPLE__)
    std::set<std::string> rpathDirs;
#endif

    for (const auto &lib : config.libraries) {
        // 按分号分割（处理 CMake 传递的分号分隔列表）
        llvm::SmallVector<llvm::StringRef, 8> libs;
        llvm::StringRef(lib).split(libs, ';');
        
        for (auto segment : libs) {
            if (segment.empty()) continue;
            
            // 判断是完整路径还是库名
            // 路径包含 '/' 或以 .so/.a/.lib 结尾，直接传递；否则加 -l 前缀
            if (segment.find('/') != llvm::StringRef::npos ||
                segment.find('\\') != llvm::StringRef::npos ||
                segment.contains(".so") ||
                segment.contains(".a") ||
                segment.contains(".lib") ||
                segment.contains(".dylib")) {
                args.push_back(segment.str());
                
                // 只为共享库添加 rpath（静态库不需要）
#if defined(__linux__) || defined(__APPLE__)
                if (segment.contains(".so") || segment.contains(".dylib")) {
                    fs::path libPath(segment.str());
                    if (libPath.has_parent_path()) {
                        rpathDirs.insert(libPath.parent_path().string());
                    }
                }
#endif
            } else {
                args.push_back("-l" + segment.str());
            }
        }
    }

    // 添加 rpath（Linux 和 macOS）
#if defined(__linux__) || defined(__APPLE__)
    for (const auto &dir : rpathDirs) {
        args.push_back("-Wl,-rpath," + dir);
    }
#endif
    
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
                                        const std::vector<std::string> archives) {
    // 加载 manifest JSON
    ManifestParser parser;
    if (mlir::failed(ManifestParser::load(manifestPath, parser))) {
        return mlir::failure();
    }
    
    auto detected = LibraryDetector::detect(module);
    
    if (!detected.libraries.empty()) {
        llvm::errs() << "Required libraries:";
        for (const auto &lib : detected.libraries)
            llvm::errs() << " -l" << lib;
        llvm::errs() << "\n";
    }
    
    LinkerConfig config;
    config.objectFile = objectFile;
    config.outputFile = outputFile;
    config.libraries = std::vector<std::string>(detected.libraries.begin(), 
                                                 detected.libraries.end());
    
    // 添加 manifest 中配置的库依赖
    for (const auto& lib : parser.getLibraries()) {
        config.libraries.push_back(lib);
    }
    
    // 解析 archive key 为完整路径
    for (const auto& key : archives) {
        std::string archivePath;
        if (mlir::failed(parser.getArchivePath(key, archivePath))) {
            return mlir::failure();
        }
        config.archives.push_back(archivePath);
    }
    
    return Linker(config).run();
}

} // namespace ezcompile::link