# --- test/integration/CMakeLists.txt ---

# 查找 lit 可执行文件
find_program(LIT_EXE lit)
if(NOT LIT_EXE)
    message(STATUS "lit not found, tests will be skipped")
    return()
endif()
message(STATUS "Found lit: ${LIT_EXE}")

# 查找 FileCheck
find_program(FILECHECK_EXE FileCheck
    PATHS "${LLVM_TOOLS_BINARY_DIR}"
    NO_DEFAULT_PATH
)
find_program(FILECHECK_EXE FileCheck)

if(NOT FILECHECK_EXE)
    message(STATUS "FileCheck not found, tests will be skipped")
    return()
endif()
message(STATUS "Found FileCheck: ${FILECHECK_EXE}")

# 获取 ezcomp 可执行文件路径
get_target_property(EZCOMP_EXE ezcomp RUNTIME_OUTPUT_DIRECTORY)
if(NOT EZCOMP_EXE)
    set(EZCOMP_EXE ${CMAKE_BINARY_DIR}/bin)
endif()
set(EZCOMP_EXECUTABLE "${EZCOMP_EXE}/ezcomp${CMAKE_EXECUTABLE_SUFFIX}")

# 配置 lit.site.cfg.py
set(FILECHECK_EXECUTABLE "${FILECHECK_EXE}")
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    @ONLY
)

# 配置 lit 测试参数
set(LIT_ARGS
    -v
    ${CMAKE_CURRENT_BINARY_DIR}
)

# 添加测试目标
add_custom_target(check
    COMMAND ${LIT_EXE} ${LIT_ARGS}
    DEPENDS ezcomp
    COMMENT "Running EzComp tests with lit"
)

# 同时添加到 ctest
add_test(
    NAME lit-tests
    COMMAND ${LIT_EXE} ${LIT_ARGS}
)
