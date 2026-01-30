# comp 方言

> 目标：作为编译器中端的最高级方言，用于把“问题级语义”（维度/网格、未知函数、初值/边界/迭代更新、邻域访问）显式编码到 IR 中。
---

## 1. 设计原则（简要）

1. **结构化保语义**：把“时间循环”“边界更新时机”“邻域访问(stencil)”这些最值钱的结构语义，保留在 comp 层。
2. **声明与执行分离**：边界条件在 `boundary` region 里**声明为句柄**；在 `step` region 的循环里**显式应用**（每步一次）。
---

## 2. 类型与属性

### 2.1 类型

- `!comp.field<T>`：未知函数（field）句柄/对象。`T` 为元素类型（如 `f64`）。
- `!comp.boundary`：单条边界条件句柄（Dirichlet/Neumann/Robin 等未来可扩展）。

### 2.2 属性

`#comp.anchor<...>`：锚点（用于 init/boundary 的“固定维度”信息）。

`index=<int>`（网格点索引）

为避免冗余，通常不再需要 `value`（可作为 debug 信息另存）。

---

## 3. Operation 语义清单

下面按“你需要实现/使用”的顺序列出每个 op 的**作用、关键操作数/属性、语义约束**。

### 3.1 `comp.problem`

`comp.problem attributes { ... } { <body> }`

顶层问题容器。承载全局 options（mode/method/precision…），并包含问题的全部声明与求解结构。

body 内必须包含：`comp.dim`（至少一个维度）、`comp.field`（至少一个未知函数）、`comp.solve`（至少一个求解入口）。

### 3.2 `comp.dim`

`comp.dim @x domain<lower=..., upper=..., points=...>`

声明一个维度符号（如空间维 `@x`、时间维 `@t`）及其均匀网格离散域。

`points >= 2`

`upper > lower`

### 3.3 `comp.field`

`%u = comp.field @u (spaceDims=[@x, ...], timeDim=@t) : !comp.field<f64>`

声明要解的未知函数，以及空间维/时间维角色。

- `timeDim` 必须引用已声明的 `comp.dim`。不应出现在 `spaceDims` 中。
- `spaceDims` 列表中的每个维度必须引用已声明的 `comp.dim`。

### 3.4 `comp.points`

`%Nt = comp.points @t : index`

读取某个维度的网格点数（`points`）。通常用于构造循环上界。

### 3.5 `comp.coord`

`%x = comp.coord @x %ix : f64`

把离散网格索引转换为该维度上的连续坐标值（uniform grid）。

### 3.6 `comp.solve`

`comp.solve %u { <init> } boundary { <boundary-decls> } step { <step> }`

把求解流程结构化成三段：
  1) `init`：一次性初始化
  2) `boundary`：声明边界条件句柄（**不直接写入**）
  3) `step`：显式时间循环与每步更新

  - `boundary` region 内只允许出现边界“声明类 op”（如 `comp.dirichlet`），返回 `!comp.boundary` 句柄。
  - `step` region 内必须对每个时间步的写层执行 `comp.enforce_boundary`（见下）。


### 3.7 `comp.apply_init`

`comp.apply_init %u anchors=[#comp.anchor<dim=@t, index=0>] { ... }`

对某一初始时间层（通常 `t=0`）初始化 `%u`。

- region 产生 RHS（标量值或表达式），用 `comp.yield` 返回。
- `anchors` 指示哪些维度被固定。

init 通常应固定 `timeDim` 到某一层（如 index=0）。

### 3.8 `comp.dirichlet`

`%b = comp.dirichlet %u anchors=[#comp.anchor<dim=@x, side=Min, index=0>] { ... } : !comp.boundary`

声明一条 Dirichlet 边界条件（边界值给定），返回可复用的 `!comp.boundary` 句柄。

- region 产生边界值（`comp.yield` 返回）。
- 边界值可以是常量，也可以依赖时间（未来可扩展为：region 传入 `%t` 或 `%n`）。

`anchors` 必须把某个空间维固定在边界（`side=Min/Max` 或者 `index=0/N-1`）。

### 3.9 `comp.for_time`

`comp.for_time %n = <lb> to <ub> step <s> { ... }`

显式时间循环（替代“隐含 Nt-1 次”）。

常见约定：`%n` 是读层索引（t=n），写层由 `comp.update` 的 `writeTime` 明确给出（通常 `n+1`）。

### 3.10 `comp.update`

`comp.update %u atTime=%n writeTime=(%n + 1) over=[@x:1..-2] { ... }`

表达“每个时间步的迭代更新”。典型用法：只更新 interior，边界由 `enforce_boundary` 负责。

- `atTime`：读取时间层（n）
- `writeTime`：写入时间层（n+1）
- `over=[@x:1..-2]`：空间更新范围（示例表示跳过两端边界点）

- **region 内容**：
  - 用 `comp.sample` 读取邻域（stencil）。
  - 用 `arith.*` 完成数值算术。
  - 用 `comp.yield` 返回要写入的标量结果。

### 3.11 `comp.sample`

`%v = comp.sample %u (@x=%ix, @t=%n) shift<@x=-1, @t=0> : f64`

邻域访问的一等公民：表达 `u(x + Δx, t + Δt)` 这种 stencil 读。

基准坐标由 `(@x=%ix, @t=%n)` 提供。

`shift<...>` 是**常量偏移**（便于编译期分析 stencil 半径与依赖）。

`shift` 必须是编译期常量整数偏移（对均匀网格）。

### 3.12 `comp.enforce_boundary`

`comp.enforce_boundary %u using (%b0, %b1, ...) atTime=(%n + 1)`

在指定时间层把一组边界句柄应用到 `%u` 上（写边界点/面）。

应当在每次时间步更新后，对**写层**（通常 `n+1`）执行一次。

`using(...)` 里的每个值必须是 `!comp.boundary`。

`atTime` 应与同一 time-step 的 `writeTime` 对齐。

### 3.13 `comp.yield`

`comp.yield %val : f64`

region 结果返回（init/boundary/update 的 RHS）。

## 4. 完整示例

> 1D heat equation 示例：
> - PDE：`∂u/∂t = 100 * ∂²u/∂x²`
> - 初值：`u(x,0)=0`
> - 边界：`u(0,t)=10`，`u(100,t)=10`
> - 网格：`x,t` 均为 `[0,100]` 上 101 点

```mlir
comp.problem attributes {
  precision = -6,
  function = "u(x,t)",
  timeVar = "t"
} {
  // 1) 维度/网格声明
  comp.dim @x domain<lower=0, upper=100, points=101>
  comp.dim @t domain<lower=0, upper=100, points=101>

  // 2) 目标未知函数：空间维=[x]，时间维=t
  %u = comp.field @u (spaceDims=[@x], timeDim=@t) : !comp.field<f64>

  // 3) 求解流程：init / boundary / step
  comp.solve %u {

    // u(x,0)=x
    comp.apply_init %u anchors=[#comp.anchor<dim=@t, index=0>] {
    ^bb0(%ix: index):
      %x = comp.coord @x %ix : f64
      comp.yield %x : f64
    }

  } boundary {
  
    // u(0,t)=10+sin(t)
    %b0 = comp.dirichlet %u anchors=[#comp.anchor<dim=@x, index=0>] {
    ^bb0(%n: index):
      %t = comp.coord @t %n : f64
      %s = math.sin %t : f64
      %c10 = arith.constant 10.0 : f64
      %rhs = arith.addf %c10, %s : f64
      comp.yield %rhs : f64
    } : (!comp.field<f64>) -> !comp.boundary

    // u(100,t)=10
    %b1 = comp.dirichlet %u anchors=[#comp.anchor<dim=@x, index=100>] {
    ^bb0(%n: index):
      comp.yield 10.0 : f64
    } : (!comp.field<f64>) -> !comp.boundary

  } step {

    %Nt = comp.points @t : index

    // (可选但推荐) 先对 t=0 的边界做一次强制应用：
    // 这样端点处的初值与边界冲突时，边界优先级明确（边界覆盖端点）。
    comp.enforce_boundary %u using (%b0, %b1) atTime=0

    // n = 0 .. Nt-2：读 t=n，写 t=n+1
    comp.for_time %n = 0 to (%Nt - 1) step 1 {

      // 1) interior 更新：只更新 x 的内部点（1..N-2）
      comp.update %u atTime=%n writeTime=(%n + 1) over=[@x:1..-2] {
        ^bb0(%ix: index):
        
          // ---- stencil 读：u(x-1,n), u(x,n), u(x+1,n) ----
          %c = comp.sample %u (@x=%ix, @t=%n) shift<@x=0,  @t=0> : f64
          %l = comp.sample %u (@x=%ix, @t=%n) shift<@x=-1, @t=0> : f64
          %r = comp.sample %u (@x=%ix, @t=%n) shift<@x=+1, @t=0> : f64

          // 网格步长（可常量折叠）
          %dx  = arith.constant 1.0 : f64
          %dt  = arith.constant 1.0 : f64
          %dx2 = arith.mulf %dx, %dx : f64

          // lap = (l + r - 2c) / dx^2
          %two  = arith.constant 2.0 : f64
          %lr   = arith.addf %l, %r : f64
          %twoC = arith.mulf %two, %c : f64
          %num  = arith.subf %lr, %twoC : f64
          %lap  = arith.divf %num, %dx2 : f64

          // next = c + 100 * dt * lap
          %k    = arith.constant 100.0 : f64
          %kdt  = arith.mulf %k, %dt : f64
          %inc  = arith.mulf %kdt, %lap : f64
          %next = arith.addf %c, %inc : f64

          comp.yield %next : f64
      }

      // 2) 每步对写层(n+1)强制应用边界（解决“每循环一次就要更新 boundary”）
      comp.enforce_boundary %u using (%b0, %b1) atTime=(%n + 1)
    }
  }
}
```
