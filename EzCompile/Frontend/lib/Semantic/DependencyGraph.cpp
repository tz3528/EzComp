//===-- DependencyGraph.cpp ------------------------------------*- C++ -*-===//
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


#include "Semantic/DependencyGraph.h"

#include <queue>

namespace ezcompile {

mlir::FailureOr<std::vector<const EquationAST*>> EqGraph::getTopoOrder() {
	std::vector<const EquationAST*> result;

	std::queue<EqNode> q;
	for (auto & node : nodes) {
		if (node.indegree == 0) {
			q.push(node);
		}
	}

	while (!q.empty()) {
		auto node = q.front();
		q.pop();
		result.push_back(node.eq);
		for (auto next:node.succ) {
			next->indegree--;
			if (next->indegree == 0) {
				q.push(*next);
			}
		}
	}

	if (result.size() != nodes.size()) {
		return mlir::failure();
	}

	return result;
}

void outputDotGraph(const EqGraph &G,std::string DotPath) {
	if (G.nodes.empty()) return;

	std::error_code EC;
	llvm::raw_fd_ostream OS(DotPath, EC, llvm::sys::fs::OF_Text);
	if (EC) {
		llvm::errs() << "Failed to open " << DotPath << ": " << EC.message() << "\n";
		return;
	}

	// 保持你原来的调用方式（参数签名不变）
	llvm::WriteGraph(OS, &G, "dependency_graph", "Dependency Graph");
	OS.flush();
}


}

