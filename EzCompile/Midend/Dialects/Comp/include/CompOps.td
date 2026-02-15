//===-- CompOps.td -----------------------------------------*- TableGen -*-===//
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


include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "CompDialect.td"

// ----------------------------------------------------------------------------
// 基础
// ----------------------------------------------------------------------------
class Comp_Op<string mnemonic, list<Trait> traits = []>
    : Op<Comp_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// 终结符（Terminator）
//===----------------------------------------------------------------------===//

// comp.yield %v : f64
// 用于“产出值”的 region 的终结符（init/boundary/update 的 RHS regions）。
// 声明为可变参数（variadic），这样需要时也可复用为通用终结符。

def YieldOp : Comp_Op<"yield", [Terminator]> {
    let summary = "Return value(s) from a comp region.";
    let arguments = (ins Variadic<AnyType>:$operands);

    // 对齐常见的 `scf.yield` 打印行为：如果没有 operands，
    // 就只打印 `comp.yield`。
    let assemblyFormat = "($operands^ `:` type($operands))? attr-dict";
}

//===----------------------------------------------------------------------===//
// 顶层 / 声明
//===----------------------------------------------------------------------===//

// comp.problem attributes { ... } { ... }

def ProblemOp : Comp_Op<"problem", [
    IsolatedFromAbove,
    SymbolTable,
    NoTerminator
]> {
    let summary = "Top-level container for a PDE/problem.";

    let regions = (region AnyRegion:$body);

    // `attr-dict-with-keyword` 只在非空时才打印 `attributes { ... }`。
    let assemblyFormat = "attr-dict-with-keyword $body";
}

// comp.dim @x domain<lower=..., upper=..., points=...>

def DimOp : Comp_Op<"dim", [Symbol]> {
    let summary = "Declare a uniform-grid dimension.";

    // 将 domain 存为三个属性。
    let arguments = (ins
        SymbolNameAttr:$sym_name,
        F64Attr:$lower,
        F64Attr:$upper,
        I64Attr:$points,
        UnitAttr:$timeVar
    );

    let assemblyFormat = "$sym_name `domain` `<` `lower` `=` $lower `,` `upper` `=` $upper `,` `points` `=` $points `>` attr-dict";

    let hasVerifier = 1; // points>=2, upper>lower（在 C++ 里实现校验）。
}

// %u = comp.field @u (spaceDims=[@x, ...], timeDim=@t) : !comp.field<f64>

def FieldOp : Comp_Op<"field", []> {
    let summary = "Declare an unknown function (field) handle.";

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        ArrayAttr:$spaceDims,
        FlatSymbolRefAttr:$timeDim
    );

    let results = (outs Comp_FieldType:$result);

    let assemblyFormat = "$sym_name `(` `spaceDims` `=` $spaceDims `,` `timeDim` `=` $timeDim `)` attr-dict `:` type($result)";

    let hasVerifier = 1; // timeDim 必须存在；不允许出现在 spaceDims 中；所有 dims 都必须已声明。
}

// %Nt = comp.points @t : index

def PointsOp : Comp_Op<"points", [Pure]> {
    let summary = "Read grid point count for a dimension.";

    let arguments = (ins FlatSymbolRefAttr:$dim);
    let results = (outs Index:$result);

    let assemblyFormat = "$dim attr-dict `:` type($result)";
}

def CoordOp : Comp_Op<"coord"> {
    let summary = "Get coordinate value of a dimension at an index (uniform grid).";

    let arguments = (ins
        FlatSymbolRefAttr:$dim,   // @x / @t
        Index:$iv                 // %ix / %n
    );

    let results = (outs F64:$coord);

    // 示例打印： %x = comp.coord @x %ix : f64
    let assemblyFormat = "$dim $iv attr-dict `:` type($coord)";
}

//===----------------------------------------------------------------------===//
// 结构化求解（Structured solve）
//===----------------------------------------------------------------------===//

// comp.solve %u { <init> } boundary { <boundary-decls> } step { <step> }
//
// 注意：init/boundary/step 这三个 region 只是容器（不要求有 terminator）。

def SolveOp : Comp_Op<"solve", [NoTerminator]> {
    let summary = "Structured solve entry: init / boundary declarations / time-step.";

    let arguments = (ins Comp_FieldType:$field);

    let regions = (region AnyRegion:$init, AnyRegion:$boundary, AnyRegion:$step);

    let assemblyFormat = "$field $init `boundary` $boundary `step` $step attr-dict `:` type($field)";
}

// comp.apply_init %u anchors=[#comp.anchor<dim=@t, index=0>] { ... }

def ApplyInitOp : Comp_Op<"apply_init"> {
    let summary = "Initialize a field at an anchored time layer.";

    let arguments = (ins
        Comp_FieldType:$field,
        ArrayAttr:$anchors
    );

    // 产生 RHS 的 region；必须以 `comp.yield <scalar>` 结束。
    let regions = (region SizedRegion<1>:$rhs);

    let assemblyFormat = "$field `anchors` `=` $anchors $rhs attr-dict `:` type($field)";

    let hasVerifier = 1; // rhs 必须 `comp.yield` 且恰好产出 1 个值。
}

// %b = comp.dirichlet %u anchors=[#comp.anchor<dim=@x, index=0>] { ... } : !comp.boundary

def DirichletOp : Comp_Op<"dirichlet"> {
    let summary = "Declare a Dirichlet boundary-condition handle.";

    let arguments = (ins
        Comp_FieldType:$field,
        ArrayAttr:$anchors
    );

    // 产生 RHS 的 region；必须以 `comp.yield <scalar>` 结束。
    let regions = (region SizedRegion<1>:$rhs);

    let assemblyFormat = "$field `anchors` `=` $anchors $rhs attr-dict `:` type($field)";

    let hasVerifier = 1; // rhs 必须 `comp.yield` 且恰好产出 1 个值。
}

//===----------------------------------------------------------------------===//
// 时间步进 / 更新 / 采样
//===----------------------------------------------------------------------===//

// comp.for_time 0 to %ub step 1 { ^bb0(%n: index): ... }
//
// 注意：这里保持标准的 region 形式（block 参数写在 region 内）。
// 如果你想要 `comp.for_time %n = <lb> to <ub> step <s> { ... }`，需要自定义 assembly format。

def ForTimeOp : Comp_Op<"for_time", [NoTerminator]> {
    let summary = "Explicit time loop (t = n).";

    let arguments = (ins
        Index:$lb,
        Index:$ub,
        Index:$step
    );

    let regions = (region SizedRegion<1>:$body);

    let assemblyFormat = "$lb `to` $ub `step` $step $body attr-dict";

    let hasVerifier = 1; // body 应该只有 1 个 block 参数（归纳变量）。
}

// comp.update %u atTime=%n writeTime=%np1 over=[...] { ^bb0(%ix: index): ... comp.yield %v : f64 }
//
// 注意：`over` 以通用 ArrayAttr 存储（具体编码方式由你选择）。

def UpdateOp : Comp_Op<"update"> {
    let summary = "Per-time-step interior update over a spatial range.";

    let arguments = (ins
        Comp_FieldType:$field,
        Index:$atTime,
        ArrayAttr:$over
    );

    // 逐元素产生 RHS 的 region（通常为逐点更新）；必须以
    // `comp.yield <scalar>` 结束。block 参数由用户定义（例如空间索引），
    // 因此这里不对它们做约束。
    let regions = (region SizedRegion<1>:$body);

    let assemblyFormat = "$field `atTime` `=` $atTime `over` `=` $over $body attr-dict `:` type($field)";

    let hasVerifier = 1; // body 必须 `comp.yield` 且恰好产出 1 个值。
}

// %v = comp.sample %u (%ix, %n) dims=[@x, @t] shift=array<i64: 0, 0> : f64
//
// 注意：这是更利于存储（storage-friendly）的编码：
//   - `indices` 为 SSA 值（index 类型）
//   - `dims` 是由 FlatSymbolRefAttr 组成的 ArrayAttr（长度与 indices 相同）
//   - `shift` 是 DenseI64ArrayAttr（长度与 indices 相同）

def SampleOp : Comp_Op<"sample", [Pure]> {
    let summary = "Stencil read: sample a field at base coords + constant shifts.";

    let arguments = (ins
        Comp_FieldType:$field,
        Variadic<Index>:$indices,
        ArrayAttr:$dims,
        DenseI64ArrayAttr:$shift
    );

    let results = (outs AnyType:$result);

    let assemblyFormat = "$field `(` $indices `)` `dims` `=` $dims `shift` `=` $shift attr-dict `:` functional-type(operands, results)";

    let hasVerifier = 1; // 尺寸约束：indices == dims == shift。
}

// comp.call @call(%a, %b) : (t1, t2) -> f64

def Comp_CallOp : Comp_Op<"call", [Pure]> {
    let summary = "Record a scalar function-style call; lowered away later";
    let arguments = (ins
        FlatSymbolRefAttr:$callee,
        Variadic<AnyType>:$args
    );
    let results = (outs F64:$result);


    let assemblyFormat = "$callee `(` $args `)` attr-dict `:` functional-type($args, $result)";
    let hasVerifier = 1;
}
