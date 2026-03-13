# --- EzResearch/CMakeLists.txt ---

# 1. 定义库目标 (EzResearch)
add_library(EzResearch STATIC)

# 2. 处理 LLVM 组件依赖
llvm_map_components_to_libnames(llvm_libs
        Support
        Core
        AsmParser
)

# 3. 链接 LLVM 库和 MLIR
target_link_libraries(EzResearch PUBLIC ${llvm_libs})
target_link_libraries(EzResearch PUBLIC
        MLIRIR
        MLIRSupport
        MLIRPass
        MLIRTransforms
        # 方言支持
        MLIRAffineDialect
        MLIRAffineTransforms
        MLIRArithDialect
        MLIRMathDialect
        MLIRMemRefDialect
        MLIRSCFDialect
        MLIRControlFlowDialect
        MLIRFuncDialect
        MLIRLLVMDialect
        MLIRVectorDialect
        # 转换支持
        MLIRAffineToStandard
        MLIRSCFToControlFlow
        MLIRConvertToLLVMPass
        MLIRVectorToLLVM
        MLIRVectorToLLVMPass
        MLIRVectorToSCF
)

# 4. 收集头文件
target_include_directories(EzResearch PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# 5. 添加子模块
add_subdirectory(Generic)
add_subdirectory(Transforms)
