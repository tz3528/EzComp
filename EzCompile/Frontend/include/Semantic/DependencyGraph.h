//===-- DependencyGraph.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// 方程依赖图接口
// 该文件定义了方程之间的依赖关系图，用于：
// - 构建方程依赖关系有向图
// - 拓扑排序确定方程计算顺序
// - 支持图可视化输出
//
//===----------------------------------------------------------------------===//


#ifndef EZ_COMPILE_DEPENDENCY_GRAPH_H
#define EZ_COMPILE_DEPENDENCY_GRAPH_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/ADT/STLExtras.h"

#include "mlir/IR/Verifier.h"

#include "AST.h"

namespace ezcompile {

/// 方程图节点，表示一个方程及其依赖关系
struct EqNode {
	const EquationAST* eq;                    // 方程指针
	llvm::SmallVector<EqNode*, 4> succ;       // 出边：依赖当前节点的方程
	unsigned indegree = 0;                    // 入度：多少方程依赖当前方程
};

/// 方程依赖图，用于拓扑排序
struct EqGraph {
	llvm::SmallVector<EqNode, 32> nodes;
	llvm::DenseMap<const EquationAST*, unsigned> idx; // 方程指针到节点索引的映射

	void addNode(const EqNode node) {
		idx.insert({node.eq, nodes.size()});
		nodes.emplace_back(node);
	}

	void addEdge(const EquationAST* def, const EquationAST* use) {
		unsigned defIdx = idx[def];
		unsigned useIdx = idx[use];

		nodes[defIdx].succ.emplace_back(&nodes[useIdx]);
		nodes[useIdx].indegree++;
	}

	/// 获取拓扑排序结果
	mlir::FailureOr<std::vector<const EquationAST*>> getTopoOrder();
};

/// 输出依赖图到DOT文件（用于可视化）
void outputDotGraph(const EqGraph &G, std::string DotPath);

}

/// 适配EqGraph结构，使其支持LLVM GraphWriter遍历
template <>
struct llvm::GraphTraits<ezcompile::EqGraph*> {
	using GraphType = ezcompile::EqGraph*;
	using NodeType = ezcompile::EqNode;
	using NodeRef  = NodeType*;

	using ChildIteratorType = llvm::SmallVector<NodeRef, 4>::iterator;

	using base_nodes_it = llvm::SmallVectorImpl<NodeType>::iterator;
	static NodeRef nodePtr(NodeType &N) { return &N; }
	using nodes_iterator = llvm::mapped_iterator<base_nodes_it, NodeRef (*)(NodeType&)>;

	static NodeRef getEntryNode(GraphType G) {
		return G->nodes.empty() ? nullptr : &G->nodes.front();
	}

	static ChildIteratorType child_begin(NodeRef N) { return N->succ.begin(); }
	static ChildIteratorType child_end  (NodeRef N) { return N->succ.end(); }

	static nodes_iterator nodes_begin(GraphType G) {
		return llvm::map_iterator(G->nodes.begin(), &nodePtr);
	}
	static nodes_iterator nodes_end(GraphType G) {
		return llvm::map_iterator(G->nodes.end(), &nodePtr);
	}
};

/// const版本特化
template <>
struct llvm::GraphTraits<const ezcompile::EqGraph*> {
	using GraphType = const ezcompile::EqGraph*;
	using NodeType = const ezcompile::EqNode;
	using NodeRef  = NodeType*;

	using ChildIteratorType = llvm::SmallVector<ezcompile::EqNode*, 4>::const_iterator;

	using base_nodes_it = llvm::SmallVectorImpl<ezcompile::EqNode>::const_iterator;
	static NodeRef nodePtr(const ezcompile::EqNode &N) { return &N; }
	using nodes_iterator = llvm::mapped_iterator<base_nodes_it, NodeRef (*)(const ezcompile::EqNode&)>;

	static NodeRef getEntryNode(GraphType G) {
		return G->nodes.empty() ? nullptr : &G->nodes.front();
	}

	static ChildIteratorType child_begin(NodeRef N) { return N->succ.begin(); }
	static ChildIteratorType child_end  (NodeRef N) { return N->succ.end(); }

	static nodes_iterator nodes_begin(GraphType G) {
		return llvm::map_iterator(G->nodes.begin(), &nodePtr);
	}
	static nodes_iterator nodes_end(GraphType G) {
		return llvm::map_iterator(G->nodes.end(), &nodePtr);
	}
};


#endif //EZ_COMPILE_DEPENDENCY_GRAPH_H
