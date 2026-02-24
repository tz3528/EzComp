# --- EzCompile/Backend/CMakeLists.txt ---

# 1. 定义库目标 (Backend)
add_library(Backend STATIC)

# 2. LLVM 依赖
llvm_map_components_to_libnames(llvm_libs
        Support
        Core
        Target
        IRReader
)

# 3. 链接依赖
target_link_libraries(Backend PUBLIC
        ${llvm_libs}
        MLIRIR
        MLIRSupport
        MLIRTargetLLVMIRExport          # MLIR -> LLVM IR 转换
        MLIRBuiltinToLLVMIRTranslation  # builtin 方言翻译接口
        MLIRLLVMToLLVMIRTranslation     # LLVM 方言翻译接口
        MLIRExecutionEngine             # JIT 执行引擎
)

# 4. 头文件目录
target_include_directories(Backend PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# 5. 添加子模块
add_subdirectory(Driver)
add_subdirectory(Link)
add_subdirectory(Target)
add_subdirectory(Translate)