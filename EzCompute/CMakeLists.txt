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

configure_file("${EZ_RUNTIME_MANIFEST_IN}" "${EZ_RUNTIME_MANIFEST_OUT}" @ONLY)

