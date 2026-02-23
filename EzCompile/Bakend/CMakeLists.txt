# --- EzCompile/Bakend/CMakeLists.txt ---

# 1. 定义库目标 (Bakend)
add_library(Bakend STATIC)

# 2. LLVM 依赖
llvm_map_components_to_libnames(llvm_libs
        Support
        Core
        Target
        IRReader
)

# 3. 链接依赖
target_link_libraries(Bakend PUBLIC
        ${llvm_libs}
        Midend
        MLIRIR
        MLIRSupport
        MLIRTargetLLVMIRExport          # MLIR -> LLVM IR 转换
        MLIRBuiltinToLLVMIRTranslation  # builtin 方言翻译接口
        MLIRLLVMToLLVMIRTranslation     # LLVM 方言翻译接口
        MLIRExecutionEngine             # JIT 执行引擎
)

# 4. 头文件目录
target_include_directories(Bakend PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# 5. 添加子模块
add_subdirectory(Driver)
add_subdirectory(Link)
add_subdirectory(Target)
add_subdirectory(Translate)