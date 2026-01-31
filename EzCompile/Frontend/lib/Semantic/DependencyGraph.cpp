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

void outputDotGraph(const EqGraph &G) {
	if (G.nodes.empty()) return;
	llvm::WriteGraph(llvm::outs(), &G, "Dependency Graph", "Dependency Graph");
}


}

