# --- EzResearch/CMakeLists.txt ---

# 1. 定义库目标 (EzResearch)
add_library(EzResearch STATIC)

# 2. 处理 LLVM 组件依赖
llvm_map_components_to_libnames(llvm_libs
        Support
        Core
)

# 3. 链接 LLVM 库和 MLIR
target_link_libraries(EzResearch PUBLIC ${llvm_libs})
target_link_libraries(EzResearch PUBLIC
        MLIRIR
        MLIRSupport
)

# 4. 收集头文件
target_include_directories(EzResearch PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# 5. 添加子模块
add_subdirectory(Transforms)
