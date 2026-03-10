# 测试设计

## 1. 概述

测试用于验证编译器各阶段正确性，采用两种测试策略：

- **集成测试**：验证编译管线中每个降级阶段的输出正确性
- **比较测试**：端到端验证，比较实际输出与预期结果

测试框架使用 LLVM lit（集成测试）配合自定义 Python 脚本（比较测试）。

---

## 2. 集成测试

### 2.1 测试配置

测试采用 LLVM lit 框架，配置分为两层：

```mermaid
flowchart LR
    A[lit.site.cfg.py.in<br/>CMake 模板] --> B[CMake configure_file]
    B --> C[lit.site.cfg.py<br/>实际配置]
    C --> D[lit.cfg.py<br/>加载配置]
```

**lit.site.cfg.py.in**：CMake 模板文件，包含占位符（如 `@EZCOMP_EXECUTABLE@`），由 CMake 配置时替换为实际路径。

**lit.cfg.py**：运行时配置，定义替换变量（`%ezcomp`、`FileCheck`）供测试文件使用。

### 2.2 各阶段测试重点

```mermaid
flowchart LR
    A1[ASTToMLIR] --> A2[CompToBase]
    A2 --> A3[AffineToSCF]
    A3 --> A4[SCFToCF]
    A4 --> A5[BaseToLLVM]
```

**ASTToMLIR**：验证 Comp 方言操作正确生成
- ProblemOp、DimOp、FieldOp 等操作的存在与属性
- Region 结构正确性

**CompToBase**：验证降级后的基础方言结构
- memref 分配维度（ping-pong 缓冲）
- affine.for 循环嵌套结构
- memref.load/store 内存访问

**AffineToSCF / SCFToCF / BaseToLLVM**：验证后续降级正确性
- 控制流转换
- LLVM 方言生成

---

## 3. 比较测试

### 3.1 测试流程

```mermaid
flowchart TB
    subgraph EzComp路线
        A1[.comp 源文件] -->|ezcomp 编译| A2[可执行文件]
        A2 -->|运行| A3[result.h5]
    end
    
    subgraph 参考实现路线
        B1[reference.cpp] -->|g++ -O0 编译| B2[可执行文件]
        B1[reference.cpp] -->|g++ -O1 编译| B3[可执行文件]
        B1[reference.cpp] -->|g++ -O2 编译| B4[可执行文件]
        B1[reference.cpp] -->|g++ -O3 编译| B5[可执行文件]
        B2 -->|运行| B6[expected_O0.h5]
        B3 -->|运行| B7[expected_O1.h5]
        B4 -->|运行| B8[expected_O2.h5]
        B5 -->|运行| B9[expected_O3.h5]
    end
    
    A3 --> C[compare.py]
    B6 --> C
    B7 --> C
    B8 --> C
    B9 --> C
    C --> D{结果一致?}
    D -->|是| E[测试通过]
    D -->|否| F[测试失败]
    C --> G[运行时间统计]
    G --> H[性能对比图]
```

参考实现使用不同优化等级编译，验证 EzComp 编译器生成的代码在各种优化级别下结果一致，同时统计并对比运行时间。

### 3.2 数值比较方法

使用相对误差和绝对误差判断数值是否匹配：
- 相对误差：`|actual - expected| / |expected|`
- 绝对误差：`|actual - expected|`

测试时通过 `rtol` 和 `atol` 参数设置允许的误差范围，两个误差只要有一个满足条件即认为匹配。

### 3.3 测试数据组织

每个测试目录包含：
- `.comp` 源文件
- `expected_result.h5` 预期输出（参考实现生成）
- 可选 `reference.cpp` 参考实现

---

## 4. 遇到的问题与解决方案

### 4.1 输出结果存在NaN的问题

**问题**：比较测试运行后，输出结果中 NaN 的个数与非边界点数相同。这说明边界点有正常数值，即 HDF5 的输入没有问题，问题出在计算环节。

**解决方案**：检查发现原始数据参数未通过数值稳定性验证，导致迭代计算发散。修改数据参数使其满足稳定性条件后，计算结果正常。

