# --- EzCompile/Midend/Dialect/lib/CMakeLists.txt ---

add_library(CompDialect STATIC
        CompDialect.cpp
        CompOps.cpp
)

# 先生成 inc 再编译 cpp
add_dependencies(CompDialect ${COMP_TABLEGEN_TARGET})

target_link_libraries(CompDialect PUBLIC
        MLIRIR
        MLIRSupport
)

# 将方言模块链接到中端模块
target_link_libraries(Midend PUBLIC CompDialect)
