# --- EzCompile/Fronted/CMakeLists.txt ---

# 1. 收集源文件
# 假设您的实现代码在当前目录下的 lib 文件夹中 (根据之前的截图)
file(GLOB_RECURSE SRCS "lib/*.cpp")

# 2. 定义库目标 (EzFrontend)
add_library(Frontend ${SRCS})

# 3. 设置头文件路径 (PUBLIC 让链接这个库的人也能看到这些头文件)
target_include_directories(Frontend PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

# 4. 处理 LLVM 组件依赖
# 将 LLVM 组件名称映射为实际的库名称
llvm_map_components_to_libnames(llvm_libs
        Support
        Core
        AsmParser
)

# 5. 链接 LLVM 库
target_link_libraries(Frontend PUBLIC ${llvm_libs})
