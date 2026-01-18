//===-- CompDialect.td -------------------------------------*- TableGen -*-===//
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


include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

def Comp_Dialect : Dialect {
    let name = "comp";

    let cppNamespace = "::ezcompile::comp";

    let summary = "Problem-level semantics dialect (dims/grids, fields, init/boundary/step, stencil).";
    let description = [{
        Comp方言是中端的最高级方言，仅用作承接前端的抽象语法树和语义分析结果
    }];
}

//===----------------------------------------------------------------------===//
// Types
//   - !comp.field<T>
//   - !comp.boundary
//===----------------------------------------------------------------------===//

def Comp_FieldType : TypeDef<Comp_Dialect, "Field"> {
    let mnemonic = "field";
    let summary = "Unknown function (field) handle/object.";
    let parameters = (ins "Type":$elementType);
    let assemblyFormat = "`<` $elementType `>`";
}

def Comp_BoundaryType : TypeDef<Comp_Dialect, "Boundary"> {
    let mnemonic = "boundary";
    let summary = "A single boundary-condition handle.";
    let assemblyFormat = "";
}

//===----------------------------------------------------------------------===//
// Attributes
//   - #comp.anchor<...>
//===----------------------------------------------------------------------===//

def Comp_AnchorAttr : AttrDef<Comp_Dialect, "Anchor"> {
    let mnemonic = "anchor";
    let summary = "Anchor for fixing dimension information in init/boundary.";

    // 过渡方案：dim 用 FlatSymbolRefAttr，能打印成 @x / @t 这样的符号引用。
    let parameters = (ins
        "mlir::FlatSymbolRefAttr":$dim,
        "int64_t":$index
    );

    let assemblyFormat =
        "`<` `dim` `=` $dim `,` `index` `=` $index `>`";
}