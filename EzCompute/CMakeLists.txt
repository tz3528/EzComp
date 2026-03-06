# --- EzCompute/CMakeLists.txt ---

# 所有生成的运行时库都放在一个目录内
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# 根据源码的json配置，复制进工作目录中
set(EZ_RUNTIME_MANIFEST_IN  "${CMAKE_SOURCE_DIR}/EzCompute/EzComputeRuntime.manifest.json")
set(EZ_RUNTIME_MANIFEST_OUT "${CMAKE_BINARY_DIR}/EzComputeRuntime.manifest.json")

add_subdirectory(IO)
add_subdirectory(Timer)

# 传递 HDF5 相关变量到父作用域 (供 compare 测试使用)
set(EZCOMPUTE_HDF5_LIBS "${EZCOMPUTE_HDF5_LIBS}" PARENT_SCOPE)
set(EZCOMPUTE_HDF5_INCLUDE_DIRS "${EZCOMPUTE_HDF5_INCLUDE_DIRS}" PARENT_SCOPE)

configure_file("${EZ_RUNTIME_MANIFEST_IN}" "${EZ_RUNTIME_MANIFEST_OUT}" @ONLY)