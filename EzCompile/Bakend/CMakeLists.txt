# --- EzCompile/Bakend/CMakeLists.txt ---

# 1. 定义库目标 (Bakend)
add_library(Bakend STATIC)

# 2. LLVM 依赖
llvm_map_components_to_libnames(llvm_libs
    Support
    Core
    Target
)

# 3. 链接依赖
target_link_libraries(Bakend PUBLIC
    ${llvm_libs}
    Midend
    MLIRIR
    MLIRSupport
)

# 4. 头文件目录
target_include_directories(Bakend PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# 5. 源文件
target_sources(Bakend PRIVATE
    Bakend.cpp
)

# 6. 子模块
add_subdirectory(Link)
add_subdirectory(Target)
add_subdirectory(Translate)