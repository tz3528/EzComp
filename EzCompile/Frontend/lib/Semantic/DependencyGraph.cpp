//===-- DependencyGraph.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// 方程依赖图实现
// 该文件实现了方程依赖图的拓扑排序算法和图可视化功能
//
//===----------------------------------------------------------------------===//


#include <queue>

#include "Semantic/DependencyGraph.h"

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

	// 输出DOT格式
	llvm::WriteGraph(OS, &G, "dependency_graph", "Dependency Graph");
	OS.flush();
}

}

