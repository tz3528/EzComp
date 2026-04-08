//===-- PolyhedralInfo.cpp ------------------------------------ -*- C++ -*-===//
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


#include "PolyhedralInfo.h"

namespace ezresearch {

PolyhedralInfo::PolyhedralInfo(mlir::affine::AffineForOp root) {
    /// 这里要找到所有的依赖关系，对于每个操作，可以定义其是否为仿射变换
    /// 对于访存操作和算数运算，可以得到它们相对于索引的仿射关系
    /// 对于所有的访存关系，可以利用它们对应索引的仿射关系，
    /// 构建一组笛卡尔积，该笛卡尔积构成了对应维度的依赖多面体
    ///
    /// 此处分析应该有的结果是:
    /// 1、所有循环嵌套内的操作所对应的仿射变换
    /// 2、构建依赖多面体
    analyzeAffineFor(root);

    /// 构建约束系统
    /// 此处要做的内容有:
    /// 1、对于所有索引，建立线性约束不等式
    /// 2、对于依赖多面体中，存在某些笛卡尔积的两个访存实际不交，应当通过对每一维的可行域求解等式是否成立
    /// 3、对于依赖多面体，需要检查依赖双方的遍历顺序是否正确
    /// 例如 S1 : A[i] = ...
    ///     S2 : ... = A[i+1]
    /// 上面会分析出来有写后读关系，但实际上遍历顺序不满足写后读
    std::vector<Dependence> tmp;
    for (auto &dependence : dependence_polyhedron) {
        if (SolveDependenceNoEqual(dependence)) {
            tmp.emplace_back(dependence);
        }
        else if (SolveDependenceEqual(dependence)) {
            tmp.emplace_back(dependence);
        }
        Dependence candidate = dependence;
        std::swap(candidate.src_indices, candidate.dst_indices);
        std::swap(candidate.src_domain, candidate.dst_domain);
        std::swap(candidate.src_memref, candidate.dst_memref);
        if (SolveDependenceNoEqual(candidate)) {
            if (candidate.kind == DependenceKind::RAW) {
                candidate.kind = DependenceKind::WAR;
            }
            else if (candidate.kind == DependenceKind::WAR) {
                candidate.kind = DependenceKind::RAW;
            }
            candidate.is_reverse = true;
            tmp.emplace_back(candidate);
        }
    }
    std::swap(dependence_polyhedron, tmp);

    /// 计算模板半径：所有访存索引偏移的极差
    int64_t max_const = 0, min_const = 0;

    auto updateRadius = [&](const std::vector<AccessInfo> &accesses) {
        for (const auto &access : accesses) {
            for (const auto &idx : access.indices) {
                if (idx.is_affine && idx.coefficient.size() == 1) {
                    max_const = std::max(max_const, idx.constant);
                    min_const = std::min(min_const, idx.constant);
                }
            }
        }
    };

    for (const auto &[memref, accesses] : memref_reads) {
        updateRadius(accesses);
    }
    for (const auto &[memref, accesses] : memref_writes) {
        updateRadius(accesses);
    }

    R = static_cast<uint64_t>(max_const - min_const);
}

void PolyhedralInfo::analyze(mlir::Operation *op) {
    if (auto constant = llvm::dyn_cast<mlir::arith::ConstantOp>(op)) {
        analyzeContant(constant);
    }
    else if (isa<mlir::arith::AddIOp, mlir::arith::SubIOp,
            mlir::arith::MulIOp, mlir::arith::DivSIOp>(op)) {
        analyzeCompute(op);
    }
    else if (auto memref_load = llvm::dyn_cast<mlir::memref::LoadOp>(op)) {
        analyzeMemrefLoad(memref_load);
    }
    else if (auto memref_store = llvm::dyn_cast<mlir::memref::StoreOp>(op)) {
        analyzeMemrefStore(memref_store);
    }
    else if (auto affine_load = llvm::dyn_cast<mlir::affine::AffineLoadOp>(op)) {
        analyzeAffineLoad(affine_load);
    }
    else if (auto affine_store = llvm::dyn_cast<mlir::affine::AffineStoreOp>(op)) {
        analyzeAffineStore(affine_store);
    }
}

void PolyhedralInfo::analyzeAffineFor(mlir::affine::AffineForOp forOp) {
    mlir::Value iv = forOp.getInductionVar();
    uint32_t id = id_to_index.size();

    index_to_id[iv] = id;
    id_to_index[id] = iv;

    // 1. 注册 IV 的自身 AffineInfo (iv = 1 * iv)
    AffineInfo ivInfo;
    ivInfo.coefficient[id] = 1;
    value_affine_map[iv] = ivInfo;

    // 2. 记录当前 domain 的大小，方便在递归退出时“弹栈”恢复
    size_t prev_domain_size = now_domain.size();

    // 3. 解析并构造下界约束： iv - lb >= 0
    if (forOp.getLowerBoundMap().getNumResults() == 1) {
        AffineInfo lb_info = buildAffineInfoFromExpr(
            forOp.getLowerBoundMap().getResult(0), forOp.getLowerBoundOperands(), value_affine_map
        );

        AffineInfo lb_constraint;
        lb_constraint.coefficient[id] = 1;         // +iv
        lb_constraint.constant = -lb_info.constant; // -lb 的常数项

        // 减去 lb 中的变量系数 (-lb)
        for (const auto& [var, coe] : lb_info.coefficient) {
            lb_constraint.coefficient[var] -= coe;
        }

        lb_constraint.is_affine = true;
        now_domain.push_back(lb_constraint);
    }

    // 4. 解析并构造上界约束： ub - iv - 1 >= 0
    if (forOp.getUpperBoundMap().getNumResults() == 1) {
        AffineInfo ub_info = buildAffineInfoFromExpr(
            forOp.getUpperBoundMap().getResult(0), forOp.getUpperBoundOperands(), value_affine_map
        );

        AffineInfo ub_constraint;
        ub_constraint.coefficient[id] = -1;             // -iv
        ub_constraint.constant = ub_info.constant - 1;  // ub 的常数项 - 1

        // 加上 ub 中的变量系数 (+ub)
        for (const auto& [var, coe] : ub_info.coefficient) {
            ub_constraint.coefficient[var] += coe;
        }

        ub_constraint.is_affine = true;
        now_domain.push_back(ub_constraint);
    }

    // 5. 将当前 IV 压入环境，开始遍历循环体

    for (mlir::Operation &op : forOp.getBody()->without_terminator()) {
        if (auto nestedFor = llvm::dyn_cast<mlir::affine::AffineForOp>(op)) {
            analyzeAffineFor(nestedFor);
        } else {
            analyze(&op);
        }
    }

    // 6. 遍历结束，恢复状态
    now_domain.resize(prev_domain_size);
}

void PolyhedralInfo::analyzeContant(mlir::arith::ConstantOp constant_op) {
    auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(constant_op.getValue());
    if (!intAttr) {
        return;
    }

    AffineInfo info;
    info.constant = intAttr.getInt();

    value_affine_map[constant_op.getResult()] = info;
}

void PolyhedralInfo::analyzeCompute(mlir::Operation *op) {
    auto getAffineInfo = [&](mlir::Value v) -> AffineInfo {
        auto it = value_affine_map.find(v);
        if (it != value_affine_map.end()) {
            return it->second;
        }

        AffineInfo unknown;
        unknown.constant = 0;
        unknown.is_affine = false;
        return unknown;
    };

    if (auto addOp = llvm::dyn_cast<mlir::arith::AddIOp>(op)) {
        AffineInfo lhs = getAffineInfo(addOp.getLhs());
        AffineInfo rhs = getAffineInfo(addOp.getRhs());
        value_affine_map[addOp.getResult()] = lhs + rhs;
    }
    else if (auto subOp = llvm::dyn_cast<mlir::arith::SubIOp>(op)) {
        AffineInfo lhs = getAffineInfo(subOp.getLhs());
        AffineInfo rhs = getAffineInfo(subOp.getRhs());
        value_affine_map[subOp.getResult()] = lhs - rhs;
    }
    else if (auto mulOp = llvm::dyn_cast<mlir::arith::MulIOp>(op)) {
        AffineInfo lhs = getAffineInfo(mulOp.getLhs());
        AffineInfo rhs = getAffineInfo(mulOp.getRhs());
        value_affine_map[mulOp.getResult()] = lhs * rhs;
    }
    else if (auto divOp = llvm::dyn_cast<mlir::arith::DivSIOp>(op)) {
        AffineInfo lhs = getAffineInfo(divOp.getLhs());
        AffineInfo rhs = getAffineInfo(divOp.getRhs());
        value_affine_map[divOp.getResult()] = lhs / rhs;
    }
}

void PolyhedralInfo::analyzeMemrefLoad(mlir::memref::LoadOp load_op) {
    AccessInfo access;
    access.op = load_op.getOperation();
    access.memref = load_op.getMemRef();
    access.is_write = false;
    access.domain = now_domain;

    // 收集每一维下标的仿射信息
    access.indices = collectIndicesAffineInfo(load_op.getIndices());

    mlir::Value memref = access.memref;

    // load 会和之前的 write 构成 RAW 依赖候选
    auto writeIt = memref_writes.find(memref);
    if (writeIt != memref_writes.end()) {
        for (const auto &prevWrite : writeIt->second) {
            Dependence dep;
            dep.src = prevWrite.op;
            dep.dst = access.op;
            dep.src_memref = prevWrite.memref;
            dep.dst_memref = access.memref;
            dep.src_indices = prevWrite.indices;
            dep.dst_indices = access.indices;
            dep.src_domain = prevWrite.domain;
            dep.dst_domain = access.domain;
            dep.kind = DependenceKind::RAW;
            dependence_polyhedron.push_back(dep);
        }
    }

    // 记录本次 read
    memref_reads[memref].push_back(access);
}

void PolyhedralInfo::analyzeMemrefStore(mlir::memref::StoreOp store_op) {
    AccessInfo access;
    access.op = store_op.getOperation();
    access.memref = store_op.getMemRef();
    access.is_write = true;
    access.domain = now_domain;

    // 收集每一维下标的仿射信息
    access.indices = collectIndicesAffineInfo(store_op.getIndices());

    mlir::Value memref = access.memref;

    // store 和之前的 read 构成 WAR
    auto readIt = memref_reads.find(memref);
    if (readIt != memref_reads.end()) {
        for (const auto &prevRead : readIt->second) {
            Dependence dep;
            dep.src = prevRead.op;
            dep.dst = access.op;
            dep.src_memref = prevRead.memref;
            dep.dst_memref = access.memref;
            dep.src_indices = prevRead.indices;
            dep.dst_indices = access.indices;
            dep.src_domain = prevRead.domain;
            dep.dst_domain = access.domain;
            dep.kind = DependenceKind::WAR;
            dependence_polyhedron.push_back(dep);
        }
    }

    // store 和之前的 write 构成 WAW
    auto writeIt = memref_writes.find(memref);
    if (writeIt != memref_writes.end()) {
        for (const auto &prevWrite : writeIt->second) {
            Dependence dep;
            dep.src = prevWrite.op;
            dep.dst = access.op;
            dep.src_memref = prevWrite.memref;
            dep.dst_memref = access.memref;
            dep.src_indices = prevWrite.indices;
            dep.dst_indices = access.indices;
            dep.src_domain = prevWrite.domain;
            dep.dst_domain = access.domain;
            dep.kind = DependenceKind::WAW;
            dependence_polyhedron.push_back(dep);
        }
    }

    // 记录本次 write
    memref_writes[memref].push_back(access);
}

void PolyhedralInfo::analyzeAffineLoad(mlir::affine::AffineLoadOp load_op) {
    AccessInfo access;
    access.op = load_op.getOperation();
    access.memref = load_op.getMemRef();
    access.is_write = false;
    access.domain = now_domain;

    access.indices = collectAffineMapIndices(
        load_op.getAffineMap(), load_op.getMapOperands(), value_affine_map
    );

    mlir::Value memref = access.memref;

    // affine.load 与之前的 write 构成 RAW
    auto writeIt = memref_writes.find(memref);
    if (writeIt != memref_writes.end()) {
        for (const auto &prevWrite : writeIt->second) {
            Dependence dep;
            dep.src = prevWrite.op;
            dep.dst = access.op;
            dep.src_memref = prevWrite.memref;
            dep.dst_memref = access.memref;
            dep.src_indices = prevWrite.indices;
            dep.dst_indices = access.indices;
            dep.src_domain = prevWrite.domain;
            dep.dst_domain = access.domain;
            dep.kind = DependenceKind::RAW;
            dependence_polyhedron.push_back(dep);
        }
    }

    memref_reads[memref].push_back(access);
}

void PolyhedralInfo::analyzeAffineStore(mlir::affine::AffineStoreOp store_op) {
    AccessInfo access;
    access.op = store_op.getOperation();
    access.memref = store_op.getMemRef();
    access.is_write = true;
    access.domain = now_domain;

    access.indices = collectAffineMapIndices(
        store_op.getAffineMap(), store_op.getMapOperands(), value_affine_map
    );

    mlir::Value memref = access.memref;

    // affine.store 与之前的 read 构成 WAR
    auto readIt = memref_reads.find(memref);
    if (readIt != memref_reads.end()) {
        for (const auto &prevRead : readIt->second) {
            Dependence dep;
            dep.src = prevRead.op;
            dep.dst = access.op;
            dep.src_memref = prevRead.memref;
            dep.dst_memref = access.memref;
            dep.src_indices = prevRead.indices;
            dep.dst_indices = access.indices;
            dep.src_domain = prevRead.domain;
            dep.dst_domain = access.domain;
            dep.kind = DependenceKind::WAR;
            dependence_polyhedron.push_back(dep);
        }
    }

    // affine.store 与之前的 write 构成 WAW
    auto writeIt = memref_writes.find(memref);
    if (writeIt != memref_writes.end()) {
        for (const auto &prevWrite : writeIt->second) {
            Dependence dep;
            dep.src = prevWrite.op;
            dep.dst = access.op;
            dep.src_memref = prevWrite.memref;
            dep.dst_memref = access.memref;
            dep.src_indices = prevWrite.indices;
            dep.dst_indices = access.indices;
            dep.src_domain = prevWrite.domain;
            dep.dst_domain = access.domain;
            dep.kind = DependenceKind::WAW;
            dependence_polyhedron.push_back(dep);
        }
    }

    memref_writes[memref].push_back(access);
}

std::vector<AffineInfo> PolyhedralInfo::collectIndicesAffineInfo(mlir::ValueRange indices) {
    std::vector<AffineInfo> result;
    for (mlir::Value idx : indices) {
        auto it = value_affine_map.find(idx);
        if (it != value_affine_map.end()) {
            result.push_back(it->second);
        } else {
            AffineInfo unknown;
            unknown.constant = 0;
            unknown.is_affine = false;
            result.push_back(unknown);
        }
    }
    return result;
}


bool SolveDependenceNoEqual(Dependence &candidate) {
    auto src_constraint = candidate.src_domain;
    auto dst_constraint = candidate.dst_domain;

    // 1. 分别收集 src 和 dst 约束中出现的变量
    llvm::DenseSet<uint32_t> src_vars, dst_vars;
    for (const auto &c : src_constraint) {
        for (const auto &[var, coe] : c.coefficient) src_vars.insert(var);
    }
    for (const auto &c : dst_constraint) {
        for (const auto &[var, coe] : c.coefficient) dst_vars.insert(var);
    }

    // 2. 安全检查：隔离验证
    for (const auto &s : candidate.src_indices) {
        for (const auto &[var, coe] : s.coefficient) {
            if (!src_vars.contains(var)) return false;
        }
    }
    for (const auto &d : candidate.dst_indices) {
        for (const auto &[var, coe] : d.coefficient) {
            if (!dst_vars.contains(var)) return false;
        }
    }

    z3::context ctx;
    z3::solver solver(ctx);

    std::map<uint32_t, z3::expr> X_map, Y_map;
    std::vector<z3::expr> shared_X, shared_Y;

    // 3. 构造变量并区分共有变量（用于字典序比较）
    // 排序保证外层循环到内层循环的深度顺序不变
    std::vector<uint32_t> ordered_src(src_vars.begin(), src_vars.end());
    std::vector<uint32_t> ordered_dst(dst_vars.begin(), dst_vars.end());
    std::sort(ordered_src.begin(), ordered_src.end());
    std::sort(ordered_dst.begin(), ordered_dst.end());

    // 创建 src 的 X 变量
    for (auto var : ordered_src) {
        X_map.emplace(var, ctx.int_const(("x_" + std::to_string(var)).c_str()));
    }
    // 创建 dst 的 Y 变量
    for (auto var : ordered_dst) {
        Y_map.emplace(var, ctx.int_const(("y_" + std::to_string(var)).c_str()));
    }

    // 提取公共循环变量（Common Surrounding Loops）用于字典序比较
    // 只有在同一个外层循环下的变量，比较它们的数值才有意义
    for (auto var : ordered_src) {
        if (dst_vars.contains(var)) {
            shared_X.push_back(X_map.at(var));
            shared_Y.push_back(Y_map.at(var));
        }
    }
    candidate.shared_depth = shared_X.size();

    // 4. 添加字典序约束 (只比较共享的循环层级)
    if (!shared_X.empty()) {
        solver.add(lex_less(ctx, shared_X, shared_Y));
    }

    // 5. 独立施加多面体域约束（X 只受 src_constraint 限制，Y 只受 dst_constraint 限制）
    for (const auto &c : src_constraint) {
        z3::expr expr_x = ctx.int_val(c.constant);
        for (const auto &[var, coe] : c.coefficient) {
            expr_x = expr_x + ctx.int_val(coe) * X_map.at(var);
        }
        solver.add(expr_x >= 0);
    }

    for (const auto &c : dst_constraint) {
        z3::expr expr_y = ctx.int_val(c.constant);
        for (const auto &[var, coe] : c.coefficient) {
            expr_y = expr_y + ctx.int_val(coe) * Y_map.at(var);
        }
        solver.add(expr_y >= 0);
    }

    // 6. 添加访存等式约束：src[i] == dst[i]
    for (size_t i = 0; i < candidate.src_indices.size(); ++i) {
        z3::expr diff = ctx.int_val(candidate.src_indices[i].constant - candidate.dst_indices[i].constant);
        for (const auto &[var, coe] : candidate.src_indices[i].coefficient) {
            diff = diff + ctx.int_val(coe) * X_map.at(var);
        }
        for (const auto &[var, coe] : candidate.dst_indices[i].coefficient) {
            diff = diff - ctx.int_val(coe) * Y_map.at(var);
        }
        solver.add(diff == 0);
    }

    if (solver.check() == z3::sat) {
        // 依赖确实存在，开始提取距离和方向
        z3::model m = solver.get_model();

        // 初始化清空（防御性编程）
        candidate.distance_vector.clear();
        candidate.direction_vector.clear();
        candidate.is_uniform = true; // 乐观假设它是模板计算的标准常数依赖

        // 1. 提取当前模型 (Model) 中的具体距离，并映射为方向
        for (size_t i = 0; i < candidate.shared_depth; ++i) {
            int x_val = m.eval(shared_X[i]).get_numeral_int();
            int y_val = m.eval(shared_Y[i]).get_numeral_int();

            // 距离定义为：目标迭代 (dst) 减去 源迭代 (src)
            int dist = y_val - x_val;
            candidate.distance_vector.push_back(dist);

            // 根据距离直接推导方向
            if (dist > 0) {
                candidate.direction_vector.push_back(DepDirection::LESS);    // src 先发生，dst 后发生
            } else if (dist == 0) {
                candidate.direction_vector.push_back(DepDirection::EQUAL);   // 同一次迭代
            } else {
                candidate.direction_vector.push_back(DepDirection::GREATER); // 违反因果律或特定的多面体依赖
            }
        }

        // 2. 验证 "常数/均匀依赖 (Uniform Dependence)" 的强假设
        // 逻辑：向 solver 提出反问：“在当前的全部约束下，你能否找到哪怕另一组解，
        // 使得它的距离向量不等于我刚刚算出来的这个 candidate.distance_vector”
        solver.push(); // 压栈：保存当前的约束状态，避免污染后续的分析

        z3::expr not_uniform_cond = ctx.bool_val(false);
        for (size_t i = 0; i < candidate.shared_depth; ++i) {
            z3::expr current_dist = shared_Y[i] - shared_X[i];
            z3::expr dist_val = ctx.int_val(candidate.distance_vector[i]);

            // 只要任何一个维度的距离发生变化，就满足“非均匀”条件
            not_uniform_cond = not_uniform_cond || (current_dist != dist_val);
        }
        solver.add(not_uniform_cond);

        // 再次求解
        if (solver.check() == z3::sat) {
            // Z3 又找到了一组不同距离的解,说明距离是随循环变量动态变化的
            candidate.is_uniform = false;

            // 既然是非常数依赖，方向向量也可能随迭代变化，退化为 STAR
            std::fill(candidate.direction_vector.begin(),
                      candidate.direction_vector.end(),
                      DepDirection::STAR);
        }

        solver.pop(); // 出栈：撤销刚才添加的 not_uniform_cond，恢复干净的状态

        return true;
    }

    return false; // 不存在依赖
}

bool SolveDependenceEqual(Dependence &candidate) {
    auto src_constraint = candidate.src_domain;
    auto dst_constraint = candidate.dst_domain;

    // 1. 分别收集变量
    llvm::DenseSet<uint32_t> src_vars, dst_vars;
    for (const auto &c : src_constraint) {
        for (const auto &[var, coe] : c.coefficient) src_vars.insert(var);
    }
    for (const auto &c : dst_constraint) {
        for (const auto &[var, coe] : c.coefficient) dst_vars.insert(var);
    }

    // 2. 安全检查
    for (const auto &s : candidate.src_indices) {
        for (const auto &[var, coe] : s.coefficient) {
            if (!src_vars.contains(var)) return false;
        }
    }
    for (const auto &d : candidate.dst_indices) {
        for (const auto &[var, coe] : d.coefficient) {
            if (!dst_vars.contains(var)) return false;
        }
    }

    z3::context ctx;
    z3::solver solver(ctx);

    std::map<uint32_t, z3::expr> X_map, Y_map;
    std::vector<uint32_t> ordered_src(src_vars.begin(), src_vars.end());
    std::vector<uint32_t> ordered_dst(dst_vars.begin(), dst_vars.end());
    std::sort(ordered_src.begin(), ordered_src.end());
    std::sort(ordered_dst.begin(), ordered_dst.end());

    // 3. 构造 Z3 变量
    for (auto var : ordered_src) {
        X_map.emplace(var, ctx.int_const(("x_" + std::to_string(var)).c_str()));
    }
    for (auto var : ordered_dst) {
        Y_map.emplace(var, ctx.int_const(("y_" + std::to_string(var)).c_str()));
    }

    // 4. 对公共嵌套循环施加等式约束 (Loop-Independent: 公共部分必须是同一次迭代)
    size_t shared_depth = 0;
    for (auto var : ordered_src) {
        if (dst_vars.contains(var)) {
            solver.add(X_map.at(var) == Y_map.at(var));
            ++shared_depth;
        }
    }
    candidate.shared_depth = shared_depth;

    // 5. 独立施加多面体域约束
    for (const auto &c : src_constraint) {
        z3::expr expr_x = ctx.int_val(c.constant);
        for (const auto &[var, coe] : c.coefficient) {
            expr_x = expr_x + ctx.int_val(coe) * X_map.at(var);
        }
        solver.add(expr_x >= 0);
    }
    for (const auto &c : dst_constraint) {
        z3::expr expr_y = ctx.int_val(c.constant);
        for (const auto &[var, coe] : c.coefficient) {
            expr_y = expr_y + ctx.int_val(coe) * Y_map.at(var);
        }
        solver.add(expr_y >= 0);
    }

    // 6. 访存等式约束
    for (size_t i = 0; i < candidate.src_indices.size(); ++i) {
        z3::expr diff = ctx.int_val(candidate.src_indices[i].constant - candidate.dst_indices[i].constant);
        for (const auto &[var, coe] : candidate.src_indices[i].coefficient) {
            diff = diff + ctx.int_val(coe) * X_map.at(var);
        }
        for (const auto &[var, coe] : candidate.dst_indices[i].coefficient) {
            diff = diff - ctx.int_val(coe) * Y_map.at(var);
        }
        solver.add(diff == 0);
    }

    if (solver.check() == z3::sat) {
        // 所有公共索引都相等的情况下存在依赖，
        // 说明这是一个 Loop-Independent Dependence。
        candidate.distance_vector.clear();
        candidate.direction_vector.clear();
        candidate.is_uniform = true; // 必然是 Uniform

        // 距离向量必然全为 0，方向必然全为 EQUAL
        for (size_t i = 0; i < candidate.shared_depth; ++i) {
            candidate.distance_vector.push_back(0);
            candidate.direction_vector.push_back(DepDirection::EQUAL);
        }

        return true;
    }

    return false;
}

}

