# 1. 声明我们需要用到的 LLVM 基础组件
set(LLVM_LINK_COMPONENTS
        Support
        Core
        AsmParser
)

# 2. 定义可执行文件
add_llvm_executable(tools
        EzComp.cpp
)

# 3. 链接库
target_link_libraries(tools PRIVATE
        # --- 项目库 ---

        # --- 其他 MLIR 基础库 ---
        MLIROptLib
        MLIRIR
        MLIRParser
        MLIRSupport
        MLIRPass
        MLIRTransforms
)

# 4. 设置 RPATH (确保运行时能找到动态库)
set_target_properties(tools PROPERTIES
        INSTALL_RPATH_USE_LINK_PATH TRUE
        BUILD_WITH_INSTALL_RPATH TRUE
)
