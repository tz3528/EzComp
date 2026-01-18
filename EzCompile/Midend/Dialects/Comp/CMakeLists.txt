# --- EzCompile/Midend/Dialect/Comp/CMakeLists.txt ---

add_subdirectory(include)
add_subdirectory(lib)


# 先生成 inc 再编译 cpp
add_dependencies(CompDialect ${COMP_TABLEGEN_TARGET})

target_link_libraries(CompDialect PUBLIC
        MLIRIR
        MLIRSupport
)

# 将方言模块链接到中端模块
target_link_libraries(Midend PUBLIC CompDialect)

target_include_directories(CompDialect PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>      # 源码头文件
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>      # 生成文件
)