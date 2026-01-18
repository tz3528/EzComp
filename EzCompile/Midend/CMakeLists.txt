# --- EzCompile/Midend/CMakeLists.txt ---

# 1. 定义库目标 (Midend)
add_library(Midend STATIC)

# 2. 处理 LLVM 组件依赖
# 将 LLVM 组件名称映射为实际的库名称
llvm_map_components_to_libnames(llvm_libs
        Support
        Core
        AsmParser
)

# 3. 链接 LLVM 库和编译器前端
target_link_libraries(Midend PUBLIC ${llvm_libs})
target_link_libraries(Midend PUBLIC Frontend)

# 收集头文件
target_include_directories(Midend PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# 5. 添加子模块
add_subdirectory(Conversions)
add_subdirectory(Dialects)
add_subdirectory(IRGen)
add_subdirectory(Transforms)
add_subdirectory(Utils)
