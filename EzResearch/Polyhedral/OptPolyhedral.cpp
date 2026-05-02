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


#include <iostream>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"

#include "DiamondTiling.h"
#include "LoopSkewing.h"
#include "PolyhedralInfo.h"
#include "Utils/CacheUtil.h"
#include "Utils/QueryUtil.h"

namespace ezresearch {

constexpr uint64_t E = 8;

/// 基于下三角幺模变换矩阵，生成新的循环嵌套并迁移代码
/// \param builder MLIR OpBuilder
/// \param loc 代码位置 (Location)
/// \param old_loops 原始的完美循环嵌套 (从外向内排序)
/// \param inv_matrix 下三角幺模逆变换矩阵 (key: 原循环层级, value: 仿射表达式信息)
/// \return 所有新生成的循环 (下标越小越外层: [0]=最外层, [depth-1]=最内层)
std::vector<mlir::affine::AffineForOp> GenerateNewLoopNests(
    mlir::OpBuilder &builder, mlir::Location loc,
    mlir::ArrayRef<mlir::affine::AffineForOp> old_loops,
    const Matrix &inv_matrix) {

    int depth = old_loops.size();
    std::vector<mlir::affine::AffineForOp> new_loops;
    std::vector<mlir::Value> new_ivs;

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

    return new_loops;
}

mlir::LogicalResult performDiamondTiling(std::vector<mlir::affine::AffineForOp>& loopBand,
                                         std::vector<uint64_t> tilingSizes) {
    if (loopBand.empty()) return mlir::failure();
    if (loopBand.size() < tilingSizes.size()) {
        return loopBand.front().emitError("Tiling sizes count exceeds available loop depth in the band.");
    }

    int depth = tilingSizes.size();
    mlir::OpBuilder builder(loopBand.front());
    mlir::Location loc = loopBand.front().getLoc();

    // 临时存储新建的循环结构
    std::vector<mlir::affine::AffineForOp> tileLoops;
    std::vector<mlir::affine::AffineForOp> pointLoops;

    // 1. 创建 Tile Loops (外层循环)
    llvm::SmallVector<mlir::Value, 4> tileIVs;
    mlir::OpBuilder tileBuilder = builder;
    mlir::IRMapping tileMap; // 用于将旧的 IV 映射到 Tile IV，防止 Dominance 错误

    for (int i = 0; i < depth; ++i) {
        auto loop = loopBand[i];

        // 动态替换 Operands 中的旧 IV 为对应的 Tile IV
        llvm::SmallVector<mlir::Value, 4> lbOperands;
        for (auto op : loop.getLowerBoundOperands()) lbOperands.push_back(tileMap.lookupOrDefault(op));

        llvm::SmallVector<mlir::Value, 4> ubOperands;
        for (auto op : loop.getUpperBoundOperands()) ubOperands.push_back(tileMap.lookupOrDefault(op));

        auto tileLoop = tileBuilder.create<mlir::affine::AffineForOp>(
            loc,
            lbOperands, loop.getLowerBoundMap(),
            ubOperands, loop.getUpperBoundMap(),
            tilingSizes[i]
        );

        tileIVs.push_back(tileLoop.getInductionVar());
        tileLoops.push_back(tileLoop); // --- 记录新建的 Tile Loop ---

        // 注册映射：后续的 Tile Loop 如果用到这一层的旧 IV，自动替换为当前的 Tile IV
        tileMap.map(loop.getInductionVar(), tileLoop.getInductionVar());

        tileBuilder.setInsertionPointToStart(tileLoop.getBody());
    }

    // 2. 创建 Point Loops (内层循环)
    llvm::SmallVector<mlir::Value, 4> pointIVs;
    mlir::IRMapping pointMap; // 用于将旧的 IV 映射到 Point IV

    for (int i = 0; i < depth; ++i) {
        auto loop = loopBand[i];
        int64_t Ti = tilingSizes[i];
        mlir::Value it = tileIVs[i];

        // 构造内层下界 max(original_LB, it)
        unsigned numOldLBDims = loop.getLowerBoundMap().getNumDims();
        llvm::SmallVector<mlir::AffineExpr, 2> lbExprs;

        for (auto expr : loop.getLowerBoundMap().getResults()) {
            lbExprs.push_back(expr);
        }
        lbExprs.push_back(tileBuilder.getAffineDimExpr(numOldLBDims));

        llvm::SmallVector<mlir::Value, 4> lbOperands;
        for (auto op : loop.getLowerBoundOperands()) {
            lbOperands.push_back(pointMap.lookupOrDefault(op));
        }
        lbOperands.push_back(it);

        auto newLBMap = mlir::AffineMap::get(
            numOldLBDims + 1,
            loop.getLowerBoundMap().getNumSymbols(),
            lbExprs, builder.getContext());

        // 构造内层上界 min(original_UB, it + Ti)
        unsigned numOldUBDims = loop.getUpperBoundMap().getNumDims();
        llvm::SmallVector<mlir::AffineExpr, 2> ubExprs;

        for (auto expr : loop.getUpperBoundMap().getResults()) {
            ubExprs.push_back(expr);
        }
        ubExprs.push_back(tileBuilder.getAffineDimExpr(numOldUBDims) + Ti);

        llvm::SmallVector<mlir::Value, 4> ubOperands;
        for (auto op : loop.getUpperBoundOperands()) {
            ubOperands.push_back(pointMap.lookupOrDefault(op));
        }
        ubOperands.push_back(it);

        auto newUBMap = mlir::AffineMap::get(
            numOldUBDims + 1,
            loop.getUpperBoundMap().getNumSymbols(),
            ubExprs, builder.getContext());

        // 创建 Point Loop
        auto pointLoop = tileBuilder.create<mlir::affine::AffineForOp>(
            loc, lbOperands, newLBMap, ubOperands, newUBMap, 1);

        pointIVs.push_back(pointLoop.getInductionVar());
        pointLoops.push_back(pointLoop); // --- 记录新建的 Point Loop ---
        pointMap.map(loop.getInductionVar(), pointLoop.getInductionVar());

        tileBuilder.setInsertionPointToStart(pointLoop.getBody());
    }

    // 3. 替换 Body 内的残余旧变量并转移 Body
    auto innerMostLoop = loopBand[depth - 1];
    auto& innerPointBody = tileBuilder.getBlock()->getOperations();
    auto& oldBody = innerMostLoop.getBody()->getOperations();

    for (int i = 0; i < depth; ++i) {
        loopBand[i].getInductionVar().replaceAllUsesWith(pointIVs[i]);
    }

    innerPointBody.splice(innerPointBody.begin(), oldBody, oldBody.begin(), std::prev(oldBody.end()));

    // 4. 安全擦除外层循环树
    loopBand.front().getOperation()->erase();

    // 5. --- 更新输出的 loopBand ---
    // 清空已经失效的旧循环指针
    loopBand.clear();
    // 依次推入：外层的 Tile Loops -> 内层的 Point Loops
    loopBand.insert(loopBand.end(), tileLoops.begin(), tileLoops.end());
    loopBand.insert(loopBand.end(), pointLoops.begin(), pointLoops.end());

    return mlir::success();
}

using namespace mlir;
using namespace mlir::affine;

/// 辅助函数：计算 affine.for 循环的常量迭代次数 (Trip Count)
int64_t getConstantTripCount(AffineForOp forOp, MLIRContext* ctx) {
    if (forOp.hasConstantBounds()) {
        int64_t lb = forOp.getConstantLowerBound();
        int64_t ub = forOp.getConstantUpperBound();
        int64_t step = forOp.getStepAsInt();
        return (ub - lb + step - 1) / step;
    }

    AffineMap lbMap = forOp.getLowerBoundMap();
    AffineMap ubMap = forOp.getUpperBoundMap();

    if (lbMap.getNumResults() == 1 && ubMap.getNumResults() == 1) {
        if (forOp.getLowerBoundOperands() == forOp.getUpperBoundOperands()) {
            AffineExpr diffExpr = ubMap.getResult(0) - lbMap.getResult(0);
            diffExpr = simplifyAffineExpr(diffExpr, lbMap.getNumDims(), lbMap.getNumSymbols());

            if (auto constExpr = dyn_cast<AffineConstantExpr>(diffExpr)) {
                int64_t diffVal = constExpr.getValue();
                int64_t step = forOp.getStepAsInt();
                return (diffVal + step - 1) / step;
            }
        }
    }
    return -1;
}

/// 核心函数：手动实现时空波前分块与并行化
void applyWavefrontParallelization(std::vector<AffineForOp>& loops, MLIRContext* ctx) {
    if (loops.empty() || loops.size() % 2 != 0) return;

    size_t n = loops.size() / 2;
    OpBuilder builder(loops[0]);
    Location loc = loops[0].getLoc();

    // ==========================================
    // 1. 提取 Tile 循环的步长与精确 Trip Count
    // ==========================================
    std::vector<int64_t> tileSteps(n);
    std::vector<int64_t> tripCounts(n);
    std::vector<Value> oldTileIVs(n);

    int64_t maxW = 0;

    for (size_t i = 0; i < n; ++i) {
        tileSteps[i] = loops[i].getStepAsInt();
        oldTileIVs[i] = loops[i].getInductionVar();

        int64_t tc = getConstantTripCount(loops[i], ctx);
        if (tc <= 0) {
            llvm::errs() << "Error: 无法推导出第 " << i << " 维的常量循环次数！\n";
            return;
        }
        tripCounts[i] = tc;

        // W = norm_0 + norm_1 + ... + norm_n-1
        maxW += (tc - 1);
    }

    // ==========================================
    // 2. 构造新的波前循环 W 和 归一化的空间循环
    // ==========================================
    // 最外层波前时间循环 (串行)
    AffineForOp wLoop = builder.create<AffineForOp>(loc, 0, maxW + 1, 1);
    builder.setInsertionPointToStart(wLoop.getBody());
    Value W = wLoop.getInductionVar();

    std::vector<Value> normIVs;
    normIVs.push_back(W);
    std::vector<AffineForOp> newSpatialLoops;

    AffineForOp innermostNewLoop = wLoop;
    for (size_t i = 1; i < n; ++i) {
        AffineForOp spatialLoop = builder.create<AffineForOp>(loc, 0, tripCounts[i], 1);
        builder.setInsertionPointToStart(spatialLoop.getBody());
        normIVs.push_back(spatialLoop.getInductionVar());
        innermostNewLoop = spatialLoop;

        newSpatialLoops.push_back(spatialLoop);
    }

    // ==========================================
    // 3. 在最内层反解原始坐标并构建合法性检查 (If Guard)
    // ==========================================
    builder.setInsertionPointToStart(innermostNewLoop.getBody());

    // 反解 norm_0 = W - norm_1 - ... - norm_{n-1}
    AffineExpr norm0Expr = builder.getAffineDimExpr(0);
    for (size_t i = 1; i < n; ++i) {
        norm0Expr = norm0Expr - builder.getAffineDimExpr(i);
    }
    AffineMap norm0Map = AffineMap::get(n, 0, norm0Expr, ctx);
    Value norm0 = builder.create<AffineApplyOp>(loc, norm0Map, normIVs);

    // 【关键】：构建 affine.if 保护！
    // 因为切片公式算出的 norm_0 必须在 [0, tripCounts[0]-1] 的范围内
    AffineExpr d0 = builder.getAffineDimExpr(0);
    AffineExpr maxTcExpr = builder.getAffineConstantExpr(tripCounts[0] - 1);
    // 约束条件： d0 >= 0  且  maxTcExpr - d0 >= 0
    SmallVector<AffineExpr, 2> constraints = { d0, maxTcExpr - d0 };
    SmallVector<bool, 2> eqFlags = { false, false }; // false 表示不等式 (>= 0)
    IntegerSet validSet = IntegerSet::get(1, 0, constraints, eqFlags);

    AffineIfOp ifOp = builder.create<AffineIfOp>(loc, validSet, ValueRange{norm0}, /*withElseRegion=*/false);

    // 后续的所有物理坐标反推和真实计算，都必须放在 If 内部！
    builder.setInsertionPointToStart(ifOp.getThenBlock());

    // 恢复时间循环物理坐标 t = norm_0 * step_0 + lower_bound
    AffineExpr iv0Expr = builder.getAffineDimExpr(0) * tileSteps[0];
    Value newIv0 = builder.create<AffineApplyOp>(loc, AffineMap::get(1, 0, iv0Expr, ctx), ValueRange{norm0});

    std::vector<Value> reconstructedIVs(n);
    reconstructedIVs[0] = newIv0;

    // 恢复空间物理坐标
    for (size_t i = 1; i < n; ++i) {
        AffineMap lbMap = loops[i].getLowerBoundMap();
        SmallVector<Value, 4> lbOperands;
        for (Value operand : loops[i].getLowerBoundOperands()) {
            if (operand == oldTileIVs[0]) lbOperands.push_back(newIv0);
            else lbOperands.push_back(operand);
        }

        Value offset = builder.create<AffineApplyOp>(loc, lbMap, lbOperands);
        AffineExpr physExpr = builder.getAffineDimExpr(0) * tileSteps[i] + builder.getAffineDimExpr(1);
        reconstructedIVs[i] = builder.create<AffineApplyOp>(
            loc, AffineMap::get(2, 0, physExpr, ctx), ValueRange{normIVs[i], offset}
        );
    }

    // ==========================================
    // 4. 代码搬运 (Splicing) 与 变量替换
    // ==========================================
    // 提取出原来的第一个 Point 循环 (最内层的起始块)
    AffineForOp firstPointLoop = loops[n];

    // 把内层的计算逻辑搬运到我们新建的 If 保护块中
    firstPointLoop->moveBefore(ifOp.getThenBlock()->getTerminator());

    // 将原始的 Tile 迭代变量全局替换为我们重建出的物理变量
    for (size_t i = 0; i < n; ++i) {
        oldTileIVs[i].replaceAllUsesWith(reconstructedIVs[i]);
    }

    // ==========================================
    // 5. 扫尾工作与并行化处理
    // ==========================================
    // 彻底销毁包含在原外层循环中的一切旧躯壳
    loops[0].erase();

    // (可选) 更新外部数组的引用，虽然旧的 loops[1~n-1] 已经析构了
    loops[0] = wLoop;

    // 【最终目标】：对于多核 CPU，仅对整体的第二维（即空间维度的第一维）进行粗粒度并行化
    if (!newSpatialLoops.empty()) {
        LogicalResult res = mlir::affine::affineParallelize(newSpatialLoops[0]);
        if (failed(res)) {
            llvm::errs() << "Warning: 官方并行化接口在处理外层空间 Tile 时失败！\n";
        }
    }
}

struct OptPolyhedralPass : public mlir::PassWrapper<OptPolyhedralPass, mlir::OperationPass<mlir::ModuleOp> > {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OptPolyhedralPass)

    llvm::StringRef getArgument() const final { return "polyhedral"; }

    void runOnOperation() override {
        mlir::ModuleOp module = getOperation();
        mlir::MLIRContext* context = &getContext();

        uint64_t CL;
        uint64_t C;
        getTilingCacheParams(CL, C);

        auto B = CL / E; // B表示每条缓存行内的元素数
        auto M = C / E; // M表示缓存中可容纳的元素数

        /// 这里首先对多面体模型进行建模
        /// 然后考虑有哪些类型的优化
        /// 对于已知的倾斜分块而言，需要做的是选取一组满足约束条件的超平面
        /// 然后基于这组超平面，对循环索引进行变换
        module.walk([&](mlir::affine::AffineForOp for_op) {
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

            if (polyhedral_info.id_to_index.size() == 1) {
                // 此处表明循环只有一层，无需优化
                return mlir::WalkResult::advance();
            }

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
            // 它会返回所有新循环 (下标越小越外层)，旧的 for_op 在函数内部已经被安全 erase 了。
            std::vector<mlir::affine::AffineForOp> new_loops = GenerateNewLoopNests(
                builder, for_op.getLoc(), old_loops, inv_matrix);

            // ---------------------------------------------------------
            // 第五步：分块
            // ---------------------------------------------------------
            std::vector<uint64_t> tiling_size = TilingSizeSolve(new_loops, B, M);

            for (auto a:tiling_size) {
                std::cout<<a<<" ";
            }
            std::cout<<"\n";

            if (mlir::failed(performDiamondTiling(new_loops, tiling_size))) {
                signalPassFailure();
            }

            // ---------------------------------------------------------
            // 第六步：波前并行
            // ---------------------------------------------------------
            applyWavefrontParallelization(new_loops, context);

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