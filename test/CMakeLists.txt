# --- test/CMakeLists.txt ---

# 查找 Python
find_package(Python3 COMPONENTS Interpreter)

# 添加子目录
add_subdirectory(integration)
add_subdirectory(compare)
