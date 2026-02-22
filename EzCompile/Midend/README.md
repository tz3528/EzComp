# comp 方言

> 目标：作为编译器中端的最高级方言，用于把"问题级语义"（维度/网格、未知函数、初值/边界/迭代更新、邻域访问）显式编码到 IR 中。

---

## 概述

**comp 方言**是 EzComp 编译器中端的最高级方言，承接前端的抽象语法树（AST）和语义分析结果。它将科学计算领域的偏微分方程（PDE）问题以结构化的方式编码到
MLIR 中，为后续的转换、优化和代码生成提供丰富的语义信息。

comp 方言的核心价值在于：

- **保留问题级语义**：将时间循环、边界更新时机、邻域访问（stencil）等关键结构语义显式表达
- **边界条件声明式设计**：边界条件在 boundary region 中声明，降级阶段自动应用
- **支持维度抽象**：自然表达空间和时间维度及其网格离散化

---

## 1. 设计原则

### 1.1 结构化保语义

把"时间循环""边界更新时机""邻域访问(stencil)"这些最关键的结构语义，保留在 comp 层，而不是过早降级到低级 IR。这允许后续优化
pass 利用这些语义信息进行针对性的优化，如：

- 并行化分析（基于时间循环结构）
- 缓存优化（基于 stencil 访问模式）
- 边界条件特化处理

### 1.2 边界条件声明式设计

边界条件采用声明式设计模式：

- **声明阶段**（`boundary` region）：定义边界条件
- **自动应用**：降级阶段自动在每个时间步后应用边界条件

这种设计带来以下好处：

- 语义清晰：边界条件的定义与迭代逻辑分离
- 简化用户代码：用户无需手动在时间循环中插入边界更新
- 便于优化：编译器可以统一处理边界条件的应用时机

### 1.3 维度显式化

所有维度（空间、时间）都必须显式声明，包括：

- 维度符号（如 `@x`、`@t`）
- 离散域（下界、上界、网格点数）
- 维度角色（空间维、时间维）

这为后续的多维分析、并行化和代码生成提供基础信息。

---

## 2. 类型与属性

### 2.1 类型

#### `!comp.field<T>`

**描述**：未知函数（field）句柄/对象类型

**参数**：

- `T`：元素类型（如 `f64`）

**用途**：表示要求解的未知函数，如温度场 `u(x,t)`、速度场 `v(x,y,t)` 等

**示例**：

```mlir
%u = comp.field @u(spaceDims = [@x], timeDim = @t) : <f64>
```

### 2.2 属性

#### `#comp.anchor<dim, index>`

**描述**：锚点属性，用于固定维度信息

**参数**：

- `dim`：维度符号引用（`FlatSymbolRefAttr`），如 `@x`、`@t`
- `index`：固定索引（`uint64_t`）

**用途**：

- 在 `comp.apply_init` 中指定初始化的时间层（如 `index=0` 表示 t=0）
- 在边界条件中指定边界位置（如 `index=0` 表示 x 的最小边界）

**示例**：

```mlir
#comp.anchor<dim = @t, index = 0>    // 固定时间维 t=0
#comp.anchor<dim = @x, index = 0>    // 固定空间维 x 的左边界
```

#### `#comp.range<dim, lower, upper>`

**描述**：范围属性，表示维度的枚举范围

**参数**：

- `dim`：维度符号引用（`FlatSymbolRefAttr`）
- `lower`：下界（`IntegerAttr`），支持负数（如 `-1` 表示 N-1）
- `upper`：上界（`IntegerAttr`），支持负数（如 `-2` 表示 N-2）

**用途**：

- 在 `comp.update` 中指定空间更新范围
- 负数索引用于表示"从末尾偏移"的语义

**示例**：

```mlir
#comp.range<dim = @x, lower = 1 : i64, upper = -2 : i64>  // 更新范围：1 到 N-2（跳过两端边界）
#comp.range<dim = @x, lower = 0 : i64, upper = -1 : i64>  // 完整范围：0 到 N-1
```

---

## 3. Operation 语义清单

下面按"你需要实现/使用"的顺序列出每个 op 的**作用、关键操作数/属性、语义约束和使用示例**。

### 3.1 `comp.problem`

**语法**：

```mlir
comp.problem attributes { ... } { <body> }
```

**描述**：顶层问题容器。承载全局 options（mode/method/precision…），并包含问题的全部声明与求解结构。

**特征（Traits）**：

- `IsolatedFromAbove`：与外部隔离
- `SymbolTable`：符号表，包含维度和 field 符号
- `NoTerminator`：不需要终结符

**语义约束**：

- `body` region 内必须包含：
    - 至少一个 `comp.dim`（维度声明）
    - 至少一个 `comp.field`（未知函数声明）
    - 至少一个 `comp.solve`（求解入口）

**示例**：

```mlir
comp.problem attributes {
  precision = -6 : i64,
  function = "u(x,t)",
  timeVar = "t"
} {
  comp.dim @t domain<lower = 0, upper = 100, points = 101> {timeVar}
  comp.dim @x domain<lower = 0, upper = 100, points = 101>
  %u = comp.field @u(spaceDims = [@x], timeDim = @t) : <f64>
  comp.solve %u { ... } boundary { ... } step { ... }
}
```

### 3.2 `comp.dim`

**语法**：

```mlir
comp.dim @x domain<lower=..., upper=..., points=...>
comp.dim @t domain<lower=..., upper=..., points=...> {timeVar}
```

**描述**：声明一个维度符号（如空间维 `@x`、时间维 `@t`）及其均匀网格离散域。

**操作数/属性**：

- `sym_name`（`SymbolNameAttr`）：维度符号名称
- `lower`（`F64Attr`）：下界
- `upper`（`F64Attr`）：上界
- `points`（`I64Attr`）：网格点数
- `timeVar`（`UnitAttr`，可选）：标记为时间维度

**语义约束**：

- `points >= 2`：至少需要 2 个网格点
- `upper > lower`：上界必须大于下界
- 时间维度必须标记 `{timeVar}` 属性

**示例**：

```mlir
comp.dim @x domain<lower = 0, upper = 100, points = 101>  // 空间维：[0, 100]，101 点
comp.dim @t domain<lower = 0, upper = 100, points = 101> {timeVar}  // 时间维：[0, 100]，101 点
```

### 3.3 `comp.field`

**语法**：

```mlir
%u = comp.field @u(spaceDims = [@x, ...], timeDim = @t) : !comp.field<f64>
```

**描述**：声明要解的未知函数，以及空间维/时间维角色。

**操作数/属性**：

- `sym_name`（`SymbolNameAttr`）：field 符号名称
- `spaceDims`（`ArrayAttr`）：空间维度列表，每个元素是 `FlatSymbolRefAttr`
- `timeDim`（`FlatSymbolRefAttr`）：时间维度引用

**结果**：

- `result`（`!comp.field<T>`）：field 句柄

**语义约束**：

- `timeDim` 必须引用已声明的 `comp.dim`
- `timeDim` 不应出现在 `spaceDims` 中
- `spaceDims` 列表中的每个维度必须引用已声明的 `comp.dim`

**示例**：

```mlir
// 2D 问题：两个空间维，一个时间维
%v = comp.field @v(spaceDims = [@x, @y], timeDim = @t) : <f64>
```

### 3.4 `comp.points`

**语法**：

```mlir
%Nt = comp.points @t : index
```

**描述**：读取某个维度的网格点数（`points`）。通常用于构造循环上界。

**操作数/属性**：

- `dim`（`FlatSymbolRefAttr`）：维度符号引用

**结果**：

- `result`（`index`）：网格点数

**特征（Traits）**：

- `Pure`：纯函数，无副作用

**示例**：

```mlir
%Nt = comp.points @t : index  // 获取时间维的点数
%Nx = comp.points @x : index  // 获取空间维的点数
comp.for_time %n = 0 to (%Nt - 1) step 1 { ... }
```

### 3.5 `comp.coord`

**语法**：

```mlir
%x = comp.coord @x %ix : f64
```

**描述**：把离散网格索引转换为该维度上的连续坐标值（uniform grid）。

**操作数/属性**：

- `dim`（`FlatSymbolRefAttr`）：维度符号引用（如 `@x`、`@t`）
- `iv`（`index`）：网格索引

**结果**：

- `coord`（`f64`）：连续坐标值

**语义约束**：

- 索引应在有效范围内 `[0, points-1]`

**示例**：

```mlir
%ix = arith.constant 5 : index
%x = comp.coord @x %ix : f64  // 假设 @x: domain<0, 100, 101>，结果约为 5.0
```

### 3.6 `comp.solve`

**语法**：

```mlir
comp.solve %u { <init> } boundary { <boundary-decls> } step { <step> }
```

**描述**：把求解流程结构化成三段：

1. `init`：一次性初始化
2. `boundary`：声明边界条件句柄（不直接写入）
3. `step`：显式时间循环与每步更新

**操作数/属性**：

- `field`（`!comp.field<T>`）：要求解的 field

**Region**：

- `init`：初始化 region（容器，无终结符要求）
- `boundary`：边界条件声明 region（容器，无终结符要求）
- `step`：时间步进 region（容器，无终结符要求）

**特征（Traits）**：

- `NoTerminator`：不需要终结符

**语义约束**：

- `boundary` region 内只允许出现边界"声明类 op"（如 `comp.dirichlet`）
- `step` region 内只需定义迭代更新逻辑，边界条件会在降级阶段自动应用

**示例**：

```mlir
comp.solve %u {
  comp.apply_init %u anchors = [#comp.anchor<dim = @t, index = 0>] {
    ^bb0(%ix: index):
      comp.yield 0.0 : f64
  } : <f64>
} boundary {
  comp.dirichlet %u anchors = [#comp.anchor<dim = @x, index = 0>] {
    ^bb0(%n: index):
      comp.yield 10.0 : f64
  } : <f64>
  comp.dirichlet %u anchors = [#comp.anchor<dim = @x, index = 100>] {
    ^bb0(%n: index):
      comp.yield 10.0 : f64
  } : <f64>
} step {
  %Nt = comp.points @t : index
  comp.for_time %c0 to %ub step %c1 {
    ...
  }
} : <f64>
```

### 3.7 `comp.apply_init`

**语法**：

```mlir
comp.apply_init %u anchors = [#comp.anchor<dim = @t, index = 0>] { ... } : <f64>
```

**描述**：对某一初始时间层（通常 `t=0`）初始化 `%u`。

**操作数/属性**：

- `field`（`!comp.field<T>`）：要初始化的 field
- `anchors`（`ArrayAttr`）：锚点数组，指示哪些维度被固定

**Region**：

- `rhs`：产生 RHS 的 region，必须以 `comp.yield` 结束，返回标量值

**语义约束**：

- `rhs` region 必须 `comp.yield` 且恰好产出 1 个值
- init 通常应固定 `timeDim` 到某一层（如 `index=0`）

**示例**：

```mlir
// 初始化为 0
comp.apply_init %u anchors = [#comp.anchor<dim = @t, index = 0>] {
  ^bb0(%ix: index):
    comp.yield 0.0 : f64
} : <f64>
```

### 3.8 `comp.dirichlet`

**语法**：

```mlir
comp.dirichlet %u anchors = [#comp.anchor<dim = @x, index = 0>] { ... } : <f64>
```

**描述**：声明一条 Dirichlet 边界条件（边界值给定）。

**操作数/属性**：

- `field`（`!comp.field<T>`）：要应用边界条件的 field
- `anchors`（`ArrayAttr`）：锚点数组，必须把某个空间维固定在边界

**Region**：

- `rhs`：产生边界值的 region，必须以 `comp.yield` 结束

**语义约束**：

- `rhs` region 必须 `comp.yield` 且恰好产出 1 个值
- 边界值可以是常量，也可以依赖时间（region 可以接收时间索引参数）
- `anchors` 必须把某个空间维固定在边界（`index=0` 或 `index=N-1`）

**示例**：

```mlir
// 常数边界：u(0,t) = 10
comp.dirichlet %u anchors = [#comp.anchor<dim = @x, index = 0>] {
  ^bb0(%n: index):
    comp.yield 10.0 : f64
} : <f64>
```

### 3.9 `comp.for_time`

**语法**：

```mlir
comp.for_time %n = <lb> to <ub> step <s> { ... }
```

**描述**：显式时间循环（替代"隐含 Nt-1 次"）。

**操作数/属性**：

- `lb`（`index`）：循环下界
- `ub`（`index`）：循环上界
- `step`（`index`）：步长

**Region**：

- `body`：循环体，应该只有 1 个 block 参数（归纳变量）

**特征（Traits）**：

- `NoTerminator`：不需要终结符

**语义约束**：

- `body` 应该只有 1 个 block 参数（归纳变量 `%n`）
- 常见约定：`%n` 是读层索引（t=n），写层由 `comp.update` 的 `writeTime` 明确给出（通常 `n+1`）

**示例**：

```mlir
// 标准：n = 0 到 Nt-1，读 t=n，写 t=n+1
comp.for_time %n = 0 to (%Nt - 1) step 1 {
  comp.update %u atTime = %n over = [#comp.range<dim = @x, lower = 1 : i64, upper = -2 : i64>] {
    ^bb0(%ix: index):
      comp.yield 0.0 : f64
  } : <f64>
}
```

### 3.10 `comp.update`

**语法**：

```mlir
comp.update %u atTime = %n over = [#comp.range<dim = @x, lower = 1 : i64, upper = -2 : i64>] { ... } : <f64>
```

**描述**：表达"每个时间步的迭代更新"。典型用法：只更新 interior，边界条件在降级阶段自动应用。

**操作数/属性**：

- `field`（`!comp.field<T>`）：要更新的 field
- `atTime`（`index`）：读取时间层（n）
- `over`（`ArrayAttr`）：空间更新范围，使用 `#comp.range` 属性

**Region**：

- `body`：逐元素产生 RHS 的 region，必须以 `comp.yield` 结束

**语义约束**：

- `body` 必须 `comp.yield` 且恰好产出 1 个值
- block 参数由用户定义（例如空间索引），因此这里不对它们做约束
- `atTime` 应与时间循环的归纳变量对齐

**Region 内容**：

- 用 `comp.sample` 读取邻域（stencil）
- 用 `arith.*` 完成数值算术
- 用 `comp.yield` 返回要写入的标量结果

**示例**：

```mlir
// 1D 中心差分：只更新内部点，跳过边界
// 参数顺序：时间索引在前，空间索引在后
comp.update %u atTime = %n over = [#comp.range<dim = @x, lower = 1 : i64, upper = -2 : i64>] {
  ^bb0(%ix: index):
    %c = comp.sample %u(%n, %ix) dims = [@t, @x] shift = [0, 0] : (!comp.field<f64>, index, index) -> f64
    %r = comp.sample %u(%n, %ix) dims = [@t, @x] shift = [0, 1] : (!comp.field<f64>, index, index) -> f64
    %l = comp.sample %u(%n, %ix) dims = [@t, @x] shift = [0, -1] : (!comp.field<f64>, index, index) -> f64
    %lap = arith.addf %l, %r : f64
    %two = arith.constant 2.0 : f64
    %two_c = arith.mulf %two, %c : f64
    %lap = arith.subf %lap, %two_c : f64
    comp.yield %lap : f64
} : <f64>
```

### 3.11 `comp.sample`

**语法**：

```mlir
%v = comp.sample %u(%n, %ix) dims = [@t, @x] shift = [0, 0] : (!comp.field<f64>, index, index) -> f64
```

**描述**：邻域访问的一等公民：表达 `u(x + Δx, t + Δt)` 这种 stencil 读。

**操作数/属性**：

- `field`（`!comp.field<T>`）：要采样的 field
- `indices`（`Variadic<Index>`）：基准坐标的索引值（SSA 值），顺序为 `[时间索引, 空间索引...]`
- `dims`（`ArrayAttr`）：维度引用数组（`FlatSymbolRefAttr`），长度与 `indices` 相同
- `shift`（`DenseI64ArrayAttr`）：常量偏移数组，长度与 `indices` 相同

**结果**：

- `result`（`AnyType`）：采样值（通常为 `f64`）

**特征（Traits）**：

- `Pure`：纯函数，无副作用

**语义约束**：

- 尺寸约束：`indices`、`dims`、`shift` 三者长度必须相同
- `shift` 必须是编译期常量整数偏移（对均匀网格）
- 基准坐标由 `(indices, dims)` 提供
- `shift` 表示在各维度上的常量偏移

**示例**：

```mlir
// 1D 模板：u(x, t)，参数顺序为 [时间索引, 空间索引]
%c = comp.sample %u(%n, %ix) dims = [@t, @x] shift = [0, 0] : (!comp.field<f64>, index, index) -> f64
// 偏移采样：u(x+1, t)
%r = comp.sample %u(%n, %ix) dims = [@t, @x] shift = [0, 1] : (!comp.field<f64>, index, index) -> f64
```

### 3.12 `comp.yield`

**语法**：

```mlir
comp.yield %val : f64
```

**描述**：region 结果返回（init/boundary/update 的 RHS）。

**操作数/属性**：

- `operands`（`Variadic<AnyType>`）：要返回的值

**特征（Traits）**：

- `Terminator`：终结符

**语义约束**：

- 在 comp 方言中，通常只返回一个值（标量）
- 用于 `comp.apply_init`、`comp.dirichlet`、`comp.update` 等 region

**示例**：

```mlir
// 返回常量
comp.yield 0.0 : f64

// 返回计算结果
%val = arith.addf %a, %b : f64
comp.yield %val : f64

// 无操作数的情况（虽然很少见）
comp.yield
```

### 3.13 `comp.call`

**语法**：

```mlir
%result = comp.call @diff(%a, %b, %c) : (f64, f64, i64) -> f64
```

**描述**：对一个函数进行调用，在降级的时候具体实现。

**操作数/属性**：

- `callee`（`FlatSymbolRefAttr`）：被调用函数的符号引用
- `args`（`Variadic<AnyType>`）：参数数组

**结果**：

- `result`（`F64`）：返回值（目前固定为 `f64`）

**特征（Traits）**：

- `Pure`：纯函数，无副作用

**语义约束**：

- 函数调用将在降级阶段被替换为具体的实现
- 目前主要用于记录函数调用语义

**示例**：

```mlir
// 调用 diff 函数
%result = comp.call @diff(%x, %t, %1) : (f64, f64, i64) -> f64

// 调用 delta 函数
%dx = comp.call @delta(%x, %2) : (f64, i64) -> f64
```

---

## 4. 完整示例

> 2D heat equation 示例：
> - PDE：`∂u/∂t = 0.5 * (∂²u/∂x² + ∂²u/∂y²)`
> - 初值：`u(x,y,0) = 0`
> - 边界：四边固定值
> - 网格：`x,y` 为 `[0,50]` 上 51 点，`t` 为 `[0,100]` 上 101 点

```mlir
module {
  comp.problem attributes {method = "FDM", mode = "time-pde", precision = -6 : i64, timeVar = "t"} {
    comp.dim @t domain<lower = 0.000000e+00, upper = 1.000000e+02, points = 101> {timeVar}
    comp.dim @x domain<lower = 0.000000e+00, upper = 5.000000e+01, points = 51>
    comp.dim @y domain<lower = 0.000000e+00, upper = 5.000000e+01, points = 51>
    %0 = comp.field @u(spaceDims = [@x, @y], timeDim = @t) : <f64>
    comp.solve %0 {
      comp.apply_init %0 anchors = [#comp.anchor<dim = @t, index = 0>] {
      ^bb0(%arg0: index, %arg1: index):
        %1 = comp.coord @x %arg0 : f64
        %2 = comp.coord @y %arg1 : f64
        %c0 = arith.constant 0 : index
        %3 = comp.coord @t %c0 : f64
        %c0_i64 = arith.constant 0 : i64
        comp.yield %c0_i64 : i64
      } : <f64>
    } boundary {
      comp.dirichlet %0 anchors = [#comp.anchor<dim = @x, index = 0>] {
      ^bb0(%arg0: index, %arg1: index):
        %1 = comp.coord @t %arg0 : f64
        %2 = comp.coord @y %arg1 : f64
        %c0 = arith.constant 0 : index
        %3 = comp.coord @x %c0 : f64
        %c100_i64 = arith.constant 100 : i64
        comp.yield %c100_i64 : i64
      } : <f64>
      comp.dirichlet %0 anchors = [#comp.anchor<dim = @x, index = 50>] {
      ^bb0(%arg0: index, %arg1: index):
        %1 = comp.coord @t %arg0 : f64
        %2 = comp.coord @y %arg1 : f64
        %c50 = arith.constant 50 : index
        %3 = comp.coord @x %c50 : f64
        %c0_i64 = arith.constant 0 : i64
        comp.yield %c0_i64 : i64
      } : <f64>
      comp.dirichlet %0 anchors = [#comp.anchor<dim = @y, index = 0>] {
      ^bb0(%arg0: index, %arg1: index):
        %1 = comp.coord @t %arg0 : f64
        %2 = comp.coord @x %arg1 : f64
        %c0 = arith.constant 0 : index
        %3 = comp.coord @y %c0 : f64
        %c100_i64 = arith.constant 100 : i64
        comp.yield %c100_i64 : i64
      } : <f64>
      comp.dirichlet %0 anchors = [#comp.anchor<dim = @y, index = 50>] {
      ^bb0(%arg0: index, %arg1: index):
        %1 = comp.coord @t %arg0 : f64
        %2 = comp.coord @x %arg1 : f64
        %c50 = arith.constant 50 : index
        %3 = comp.coord @y %c50 : f64
        %c0_i64 = arith.constant 0 : i64
        comp.yield %c0_i64 : i64
      } : <f64>
    } step {
      %1 = comp.points @t : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %2 = arith.subi %1, %c1 : index
      comp.for_time %c0 to %2 step %c1 {
      ^bb0(%arg0: index):
        %3 = comp.coord @t %arg0 : f64
        comp.update %0 atTime = %arg0 over = [#comp.range<dim = @x, lower = -1 : i64, upper = 1 : i64>, #comp.range<dim = @y, lower = -1 : i64, upper = 1 : i64>] {
        ^bb0(%arg1: index, %arg2: index):
          %4 = comp.coord @x %arg1 : f64
          %5 = comp.coord @y %arg2 : f64
          %6 = comp.sample %0(%arg0, %arg1, %arg2) dims = [@t, @x, @y] shift = [0, -1, 0] : (!comp.field<f64>, index, index, index) -> f64
          %7 = comp.sample %0(%arg0, %arg1, %arg2) dims = [@t, @x, @y] shift = [0, 0, -1] : (!comp.field<f64>, index, index, index) -> f64
          %8 = comp.sample %0(%arg0, %arg1, %arg2) dims = [@t, @x, @y] shift = [0, 0, 0] : (!comp.field<f64>, index, index, index) -> f64
          %9 = comp.sample %0(%arg0, %arg1, %arg2) dims = [@t, @x, @y] shift = [0, 0, 1] : (!comp.field<f64>, index, index, index) -> f64
          %10 = comp.sample %0(%arg0, %arg1, %arg2) dims = [@t, @x, @y] shift = [0, 1, 0] : (!comp.field<f64>, index, index, index) -> f64
          %11 = comp.sample %0(%arg0, %arg1, %arg2) dims = [@t, @x, @y] shift = [1, 0, 0] : (!comp.field<f64>, index, index, index) -> f64
          %12 = arith.addf %6, %10 : f64
          %c2_i64 = arith.constant 2 : i64
          %13 = arith.sitofp %c2_i64 : i64 to f64
          %14 = arith.mulf %13, %8 : f64
          %15 = arith.subf %12, %14 : f64
          %c2_i64_0 = arith.constant 2 : i64
          %16 = comp.call @delta(%4, %c2_i64_0) : (f64, i64) -> f64
          %17 = arith.divf %15, %16 : f64
          %18 = arith.addf %7, %9 : f64
          %c2_i64_1 = arith.constant 2 : i64
          %19 = arith.sitofp %c2_i64_1 : i64 to f64
          %20 = arith.mulf %19, %8 : f64
          %21 = arith.subf %18, %20 : f64
          %c2_i64_2 = arith.constant 2 : i64
          %22 = comp.call @delta(%5, %c2_i64_2) : (f64, i64) -> f64
          %23 = arith.divf %21, %22 : f64
          %cst = arith.constant 5.000000e-01 : f64
          %24 = arith.addf %17, %23 : f64
          %25 = arith.mulf %cst, %24 : f64
          %c1_i64 = arith.constant 1 : i64
          %26 = comp.call @delta(%3, %c1_i64) : (f64, i64) -> f64
          %27 = arith.mulf %25, %26 : f64
          %28 = arith.addf %8, %27 : f64
          comp.yield %28 : f64
        } : <f64>
      }
    } : <f64>
  }
}
```
