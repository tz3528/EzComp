//===-- DiagnosticBase.h ---------------------------------------*- C++ -*-===//
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


#ifndef EZ_COMPILE_DIAGNOSTIC_BASE_H
#define EZ_COMPILE_DIAGNOSTIC_BASE_H

#include "llvm/Support/SourceMgr.h"
#include "llvm/ADT/Twine.h"

namespace ezcompile {

/// 诊断基类
/// 提供 sourceMgr 引用和统一的 emitError 接口
class DiagnosticBase {
public:
	explicit DiagnosticBase(llvm::SourceMgr &sourceMgr, int bufferID, mlir::MLIRContext *ctx)
	  : sourceMgr(sourceMgr), bufferID(bufferID), ctx(ctx) {}

	/// 检查是否有错误发生
	bool hadError() const { return hasError; }

protected:
	/// 统一报错接口
	void emitError(llvm::SMLoc loc, llvm::StringRef msg) {
		hasError = true;

		if (!ctx) {
			sourceMgr.PrintMessage(loc, llvm::SourceMgr::DK_Error, msg);
			return;
		}

		auto lineCol = sourceMgr.getLineAndColumn(loc, bufferID);
		auto *buf = sourceMgr.getMemoryBuffer(bufferID);
		llvm::StringRef file = buf ? buf->getBufferIdentifier() : "<input>";
		auto flc = mlir::FileLineColLoc::get(ctx, file, lineCol.first, lineCol.second);
		mlir::emitError(flc) << msg;
	}

private:
	llvm::SourceMgr &sourceMgr;
	int bufferID;
	mlir::MLIRContext *ctx = nullptr;
	bool hasError = false;
};

}

#endif //EZ_COMPILE_DIAGNOSTIC_BASE_H
