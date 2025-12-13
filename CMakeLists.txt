cmake_minimum_required(VERSION 3.28.3)

# 1. 定义项目名称
project(EzComp LANGUAGES CXX C)

# ---------------------------------------------------------------------------
# 2. C++ 标准设置 (你要求的 C++20)
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# 3. 寻找依赖：MLIR 和 LLVM
# ---------------------------------------------------------------------------
# 编译时需要指定 -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir
find_package(MLIR REQUIRED CONFIG)

# ---------------------------------------------------------------------------
# 4. 加载 MLIR/LLVM 的 CMake 辅助工具
# ---------------------------------------------------------------------------
# 这一步是为了让我们能使用 add_mlir_dialect, mlir_tablegen 等官方宏
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions) # 同步 LLVM 的编译选项 (如 RTTI, Exceptions 等)

# ---------------------------------------------------------------------------
# 5. 设置头文件路径
# ---------------------------------------------------------------------------
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
# 让项目内的代码可以互相引用 (比如 #include "EzCompile/...")
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
# 让项目可以引用编译生成的中间文件 (比如 TableGen 生成的 .inc 文件)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# ---------------------------------------------------------------------------
# 6. 添加子目录
# ---------------------------------------------------------------------------
add_subdirectory(EzCompile)
add_subdirectory(EzCompute)
add_subdirectory(tools)
