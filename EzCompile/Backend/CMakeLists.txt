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

# 6. 设置 Clang++ 路径（优先使用 LLVM 构建目录中的 clang）
# LLVM_TOOLS_BINARY_DIR 是 LLVM 工具（如 clang++）的输出目录
if(DEFINED LLVM_TOOLS_BINARY_DIR)
    set(CLANGPLUSPLUS_PATH "${LLVM_TOOLS_BINARY_DIR}/clang++${CMAKE_EXECUTABLE_SUFFIX}")
elseif(DEFINED LLVM_BINARY_DIR)
    set(CLANGPLUSPLUS_PATH "${LLVM_BINARY_DIR}/bin/clang++${CMAKE_EXECUTABLE_SUFFIX}")
else()
    # Fallback: 相对于 MLIR_DIR 推算
    get_filename_component(LLVM_BUILD_DIR "${MLIR_DIR}/../../.." ABSOLUTE)
    set(CLANGPLUSPLUS_PATH "${LLVM_BUILD_DIR}/bin/clang++${CMAKE_EXECUTABLE_SUFFIX}")
endif()

target_compile_definitions(Backend PRIVATE
        CLANGPLUSPLUS_PATH="${CLANGPLUSPLUS_PATH}"
)

# 6. 添加子模块
add_subdirectory(Driver)
add_subdirectory(Link)
add_subdirectory(Target)
add_subdirectory(Translate)