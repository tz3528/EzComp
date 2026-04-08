//===-- OptPolyhedral.cpp ------------------------------------- -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
///
//===----------------------------------------------------------------------===//


#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"

#include "LoopSkewing.h"
#include "PolyhedralInfo.h"
#include "Utils/QueryUtil.h"

namespace ezresearch {

/// 基于下三角幺模变换矩阵，生成新的循环嵌套并迁移代码
/// \param builder MLIR OpBuilder
/// \param loc 代码位置 (Location)
/// \param old_loops 原始的完美循环嵌套 (从外向内排序)
/// \param inv_matrix 下三角幺模逆变换矩阵 (key: 原循环层级, value: 仿射表达式信息)
/// \return 新的最外层循环
mlir::affine::AffineForOp GenerateNewLoopNests(
    mlir::OpBuilder &builder, mlir::Location loc,
    mlir::ArrayRef<mlir::affine::AffineForOp> old_loops,
    const Matrix &inv_matrix) {

    int depth = old_loops.size();
    mlir::SmallVector<mlir::affine::AffineForOp, 4> new_loops;
    mlir::SmallVector<mlir::Value, 4> new_ivs;

    // 用于存储：旧的外层 IV 的 AffineExpr 如何被新的外层 IV 的 AffineExpr 表示
    mlir::SmallVector<mlir::AffineExpr, 4> old_iv_exprs_in_new_ivs;

    // ==========================================================
    // 第一阶段：逐层计算新边界，并创建新的 AffineForOp 嵌套
    // ==========================================================
    for (int k = 0; k < depth; ++k) {
        mlir::affine::AffineForOp old_loop = old_loops[k];
        const AffineInfo &inv_info = inv_matrix.at(k);

        // 1. 获取旧的边界映射
        mlir::AffineMap old_lb_map = old_loop.getLowerBoundMap();
        mlir::AffineMap old_ub_map = old_loop.getUpperBoundMap();

        // 2. 将旧边界中的旧变量(Dims)替换为已推导出的新变量表达式
        mlir::AffineMap sub_lb_map = old_lb_map.replaceDimsAndSymbols(
            old_iv_exprs_in_new_ivs, /*symbols=*/{},
            old_lb_map.getNumDims(), old_lb_map.getNumSymbols());
        mlir::AffineMap sub_ub_map = old_ub_map.replaceDimsAndSymbols(
            old_iv_exprs_in_new_ivs, /*symbols=*/{},
            old_ub_map.getNumDims(), old_ub_map.getNumSymbols());

        // 3. 构建当前层旧变量 i_k 中除了 u_k 以外的 "Rest" 表达式
        mlir::AffineExpr rest_expr = builder.getAffineConstantExpr(inv_info.constant);
        int64_t uk_coeff = 0;

        for (const auto &pair : inv_info.coefficient) {
            uint32_t new_dim_idx = pair.first;
            int64_t coeff = pair.second;

            if (new_dim_idx == k) {
                uk_coeff = coeff;
            } else {
                // 下三角性质保证：这里出现的 new_dim_idx 必然严格小于 k
                mlir::AffineExpr dim_expr = builder.getAffineDimExpr(new_dim_idx);
                rest_expr = rest_expr + dim_expr * coeff;
            }
        }

        // 幺模矩阵性质：对角线必须是 1 或 -1
        assert((uk_coeff == 1 || uk_coeff == -1) && "Unimodular lower triangular matrix diagonal must be +/-1");

        // 4. 根据 uk_coeff 的符号推导 u_k 的新边界表达式
        mlir::SmallVector<mlir::AffineExpr, 4> new_lb_exprs, new_ub_exprs;
        if (uk_coeff == 1) { // u_k >= L - Rest; u_k <= U - Rest
            for (mlir::AffineExpr expr : sub_lb_map.getResults())
                new_lb_exprs.push_back(expr - rest_expr);
            for (mlir::AffineExpr expr : sub_ub_map.getResults())
                new_ub_exprs.push_back(expr - rest_expr);
        } else { // uk_coeff == -1 -> 边界翻转: u_k >= Rest - U; u_k <= Rest - L
            for (mlir::AffineExpr expr : sub_ub_map.getResults())
                new_lb_exprs.push_back(rest_expr - expr);
            for (mlir::AffineExpr expr : sub_lb_map.getResults())
                new_ub_exprs.push_back(rest_expr - expr);
        }

        // 构建新的 AffineMap (维度数量为 k，即依赖外层的 k 个新变量)
        mlir::AffineMap new_lb_map = mlir::AffineMap::get(k, sub_lb_map.getNumSymbols(), new_lb_exprs, builder.getContext());
        mlir::AffineMap new_ub_map = mlir::AffineMap::get(k, sub_ub_map.getNumSymbols(), new_ub_exprs, builder.getContext());

        // 5. 组装新循环的 Operands (由外层的新 IVs + 旧循环中原有的 Symbols 组成)
        mlir::SmallVector<mlir::Value, 4> lb_operands(new_ivs.begin(), new_ivs.end());
        auto old_lb_syms = old_loop.getLowerBoundOperands().drop_front(old_lb_map.getNumDims());
        lb_operands.append(old_lb_syms.begin(), old_lb_syms.end());

        mlir::SmallVector<mlir::Value, 4> ub_operands(new_ivs.begin(), new_ivs.end());
        auto old_ub_syms = old_loop.getUpperBoundOperands().drop_front(old_ub_map.getNumDims());
        ub_operands.append(old_ub_syms.begin(), old_ub_syms.end());

        // 6. 创建当前层的新循环
        auto new_loop = builder.create<mlir::affine::AffineForOp>(
            loc, lb_operands, new_lb_map, ub_operands, new_ub_map);

        new_loops.push_back(new_loop);
        new_ivs.push_back(new_loop.getInductionVar());

        // 7. 将 i_k 的完整表达式存入表，供更内层循环 (k+1...) 替换使用
        mlir::AffineExpr ik_expr = rest_expr + builder.getAffineDimExpr(k) * uk_coeff;
        old_iv_exprs_in_new_ivs.push_back(ik_expr);

        // 移动 builder 插入点，准备生成下一层循环
        builder.setInsertionPointToStart(new_loop.getBody());
    }

    // ==========================================================
    // 第二阶段：转移循环体并替换变量引用
    // ==========================================================

    mlir::affine::AffineForOp old_inner_loop = old_loops.back();
    mlir::Block *old_inner_body = old_inner_loop.getBody();
    mlir::Block *new_inner_body = new_loops.back().getBody();

    // 1. 物理转移：将旧最内层操作全部 Splice 移至新最内层 (跳过最后自带的 affine.yield)
    auto &old_ops = old_inner_body->getOperations();
    auto &new_ops = new_inner_body->getOperations();
    new_ops.splice(std::prev(new_ops.end()), old_ops, old_ops.begin(), std::prev(old_ops.end()));

    // 2. 逻辑替换：在新循环体内插入 affine.apply，算出旧 IV 值，并全局替换
    builder.setInsertionPointToStart(new_inner_body);

    for (int i = 0; i < depth; ++i) {
        mlir::affine::AffineForOp old_loop_i = old_loops[i];
        mlir::Value old_iv = old_loop_i.getInductionVar();
        const AffineInfo &info = inv_matrix.at(i);

        // 基于逆矩阵重构仿射表达式
        mlir::AffineExpr expr = builder.getAffineConstantExpr(info.constant);
        for (const auto &pair : info.coefficient) {
            expr = expr + builder.getAffineDimExpr(pair.first) * pair.second;
        }

        // 创建 affine.apply 操作 (输入为所有的新 IV)
        mlir::AffineMap apply_map = mlir::AffineMap::get(depth, 0, expr, builder.getContext());
        auto apply_op = builder.create<mlir::affine::AffineApplyOp>(loc, apply_map, new_ivs);

        // 替换 Body 内所有对原变量的使用
        old_iv.replaceAllUsesWith(apply_op.getResult());
    }

    // ==========================================================
    // 第三阶段：清理旧的循环树
    // ==========================================================
    mlir::affine::AffineForOp outermost_old_loop = old_loops.front();
    outermost_old_loop.erase();

    return new_loops.front();
}

struct OptPolyhedralPass : public mlir::PassWrapper<OptPolyhedralPass, mlir::OperationPass<mlir::ModuleOp> > {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OptPolyhedralPass)

    llvm::StringRef getArgument() const final { return "polyhedral"; }

    void runOnOperation() override {
        mlir::ModuleOp module = getOperation();

        /// 这里首先对多面体模型进行建模
        /// 然后考虑有哪些类型的优化
        /// 对于已知的倾斜分块而言，需要做的是选取一组满足约束条件的超平面
        /// 然后基于这组超平面，对循环索引进行变换

        module.walk([](mlir::affine::AffineForOp for_op) {
            // 过滤：只抓取最外层（根部）循环
            if (for_op->getParentOfType<mlir::affine::AffineForOp>()) {
                return mlir::WalkResult::advance();
            }

            // 检查：如果这个独立的嵌套树不是完美的，跳过它，去处理下一个顶层循环
            if (!isPerfectLoopNest(for_op)) {
                // 这里可以用 emitRemark 或 emitWarning，而不是抛出 Error 中断整个 Pass
                for_op.emitRemark("Skipping non-perfectly nested loop tree.");
                return mlir::WalkResult::advance();
            }

            PolyhedralInfo polyhedral_info(for_op);

            auto matrix = SolveSkewingMatrix(polyhedral_info);

            // ---------------------------------------------------------
            // 第二步：获取旧循环的层级信息与归纳变量 (IVs)
            // ---------------------------------------------------------
            llvm::SmallVector<mlir::affine::AffineForOp, 4> old_loops;
            mlir::affine::getPerfectlyNestedLoops(old_loops, for_op);

            llvm::SmallVector<mlir::Value, 4> old_ivs;
            for (auto loop : old_loops) {
                old_ivs.push_back(loop.getInductionVar());
            }

            // ---------------------------------------------------------
            // 第三步：计算逆矩阵 T^{-1} (适配 std::map<uint32_t, AffineInfo> 格式)
            // ---------------------------------------------------------
            Matrix inv_matrix = getInversionMatrix(matrix);

            // ---------------------------------------------------------
            // 第四步：生成新循环并完成所有代码转移与清理
            // ---------------------------------------------------------
            // 初始化 builder，并将插入点设置在原本的旧最外层循环之前
            mlir::OpBuilder builder(for_op);

            // 直接调用我们写好的全包函数！
            // 它会返回新的最外层循环，旧的 for_op 在函数内部已经被安全 erase 了。
            mlir::affine::AffineForOp new_outer_loop = GenerateNewLoopNests(
                builder, for_op.getLoc(), old_loops, inv_matrix);

            // (注意：这里之后不要再对 for_op 或 old_loops 进行任何操作，它们已经被销毁了)

            return mlir::WalkResult::advance();

        });
    }
};

void registerOptPolyhedralPass() {
    mlir::PassRegistration<OptPolyhedralPass>();
}

std::unique_ptr<mlir::Pass> createOptPolyhedralPass() {
    return std::make_unique<OptPolyhedralPass>();
}

}