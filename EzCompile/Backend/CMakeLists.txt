# --- EzCompile/Backend/CMakeLists.txt ---

# 1. 定义库目标 (Backend)
add_library(Backend STATIC)

# 2. LLVM 依赖
# LLVM_TARGETS_TO_BUILD: LLVM 编译时配置的目标后端列表
llvm_map_components_to_libnames(llvm_libs
        Support
        Core
        Target
        IRReader
        ${LLVM_TARGETS_TO_BUILD}
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

# 5. 设置json配置文件的路径
target_compile_definitions(Backend PRIVATE
        EZ_RUNTIME_MANIFEST_ABS_PATH="${CMAKE_BINARY_DIR}/EzComputeRuntime.manifest.json"
)

# 6. 设置 C++ 编译器路径（使用编译项目的编译器）
target_compile_definitions(Backend PRIVATE
        EZCOMP_CXX_COMPILER="${CMAKE_CXX_COMPILER}"
)

# 6. 添加子模块
add_subdirectory(Driver)
add_subdirectory(Link)
add_subdirectory(Target)
add_subdirectory(Translate)