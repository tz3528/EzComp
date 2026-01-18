# --- EzCompile/Midend/Dialect/Comp/include/CMakeLists.txt ---

set(LLVM_TARGET_DEFINITIONS CompDialect.td)
mlir_tablegen(CompDialect.h.inc -gen-dialect-decls -dialect=comp)
mlir_tablegen(CompDialect.cpp.inc -gen-dialect-defs -dialect=comp)

mlir_tablegen(CompTypes.h.inc  -gen-typedef-decls -typedefs-dialect=comp)
mlir_tablegen(CompTypes.cpp.inc -gen-typedef-defs  -typedefs-dialect=comp)

mlir_tablegen(CompAttributes.h.inc  -gen-attrdef-decls -attrdefs-dialect=comp)
mlir_tablegen(CompAttributes.cpp.inc -gen-attrdef-defs  -attrdefs-dialect=comp)

add_public_tablegen_target(CompDialectIncGen)

set(LLVM_TARGET_DEFINITIONS CompOps.td)
mlir_tablegen(CompOps.h.inc -gen-op-decls -dialect=comp)
mlir_tablegen(CompOps.cpp.inc -gen-op-defs -dialect=comp)
add_public_tablegen_target(CompOpsIncGen)

set(COMP_TABLEGEN_TARGET
        CompDialectIncGen
        CompOpsIncGen
        PARENT_SCOPE
)

set(COMP_PUBLIC_HEADERS
        CompDialect.h
        CompOps.h
        PARENT_SCOPE
)