# --- test/compare/CMakeLists.txt ---

# 检查 Python 包依赖
if(Python3_Interpreter_FOUND)
    # 检查 h5py
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import h5py"
        RESULT_VARIABLE H5PY_CHECK
        OUTPUT_QUIET ERROR_QUIET
    )
    if(NOT H5PY_CHECK EQUAL 0)
        message(WARNING "h5py not found, compare tests may fail. Install with: pip install h5py")
    endif()

    # 检查 matplotlib
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import matplotlib"
        RESULT_VARIABLE MATPLOTLIB_CHECK
        OUTPUT_QUIET ERROR_QUIET
    )
    if(NOT MATPLOTLIB_CHECK EQUAL 0)
        message(WARNING "matplotlib not found, performance plot will not be generated. Install with: pip install matplotlib")
    endif()
endif()

# 获取 ezcomp 可执行文件路径
get_target_property(EZCOMP_EXE ezcomp RUNTIME_OUTPUT_DIRECTORY)
if(NOT EZCOMP_EXE)
    set(EZCOMP_EXE ${CMAKE_BINARY_DIR}/bin)
endif()
set(EZCOMP_EXECUTABLE "${EZCOMP_EXE}/ezcomp${CMAKE_EXECUTABLE_SUFFIX}")

# 获取 C++ 编译器路径
set(CXX_COMPILER "${CMAKE_CXX_COMPILER}")

# HDF5 库和头文件路径 (从 EzCompute 传递)
set(HDF5_INCLUDE_DIRS "${EZCOMPUTE_HDF5_INCLUDE_DIRS}")

# 测试数据源码目录
set(COMPARE_TEST_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

# 配置比较测试脚本 (注入 ezcomp 路径、编译器、HDF5 信息和源码目录)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/compare.py
    ${CMAKE_CURRENT_BINARY_DIR}/compare.py
    @ONLY
)

# 添加 compare target
# 用法: COMPARE_TEST_DIR=<test_dir> cmake --build . --target compare
# 例如: COMPARE_TEST_DIR=2d_heat_equation cmake --build . --target compare
add_custom_target(compare
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/compare.py $ENV{COMPARE_TEST_DIR}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running compare script for test: $ENV{COMPARE_TEST_DIR}"
    USES_TERMINAL
)