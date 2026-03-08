# --- EzCompute/IO/CMakeLists.txt ---

# 默认使用系统 HDF5，设为 ON 则从子模块编译
set(EZCOMP_BUILD_HDF5_SUBMODULE OFF CACHE BOOL "Build HDF5 from submodule instead of using system")

add_library(HDF5Wrapper STATIC)

target_sources(HDF5Wrapper PRIVATE
        HDF5Wrapper.cpp
)

# 使用跨平台变量处理库路径
set(EZCOMPUTE_HDF5_WRAPPER_REL  "lib/${CMAKE_STATIC_LIBRARY_PREFIX}HDF5Wrapper${CMAKE_STATIC_LIBRARY_SUFFIX}" PARENT_SCOPE)

if(EZCOMP_BUILD_HDF5_SUBMODULE)
    # 从子模块编译 HDF5
    message(STATUS "Building HDF5 from submodule")
    
    # HDF5 配置选项（必须在 add_subdirectory 之前设置）
    set(HDF5_EXTERNALLY_CONFIGURED ON CACHE INTERNAL "")
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build Shared Libraries" FORCE)
    set(BUILD_STATIC_LIBS ON CACHE BOOL "Build Static Libraries" FORCE)
    set(HDF5_ENABLE_PARALLEL OFF CACHE INTERNAL "")
    set(HDF5_ENABLE_ZLIB_SUPPORT OFF CACHE INTERNAL "")
    
    # 禁用不必要的组件
    set(BUILD_TESTING OFF CACHE INTERNAL "")
    set(HDF5_BUILD_TOOLS OFF CACHE INTERNAL "")
    set(HDF5_BUILD_EXAMPLES OFF CACHE INTERNAL "")
    set(HDF5_BUILD_UTILS OFF CACHE INTERNAL "")
    set(HDF5_BUILD_HL_LIB OFF CACHE INTERNAL "")
    
    # 添加 HDF5 子目录
    add_subdirectory(hdf5 hdf5-build)
    
    # HDF5 输出目录设置
    set(HDF5_OUTPUT_DIR "${CMAKE_BINARY_DIR}/lib")
    set_target_properties(hdf5-static PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${HDF5_OUTPUT_DIR}"
    )
    
    # 设置 HDF5 库路径
    set(HDF5_STATIC_LIB "${HDF5_OUTPUT_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}hdf5${CMAKE_STATIC_LIBRARY_SUFFIX}")
    set(EZCOMPUTE_HDF5_LIBS "${HDF5_STATIC_LIB}" PARENT_SCOPE)
    set(EZCOMPUTE_HDF5_INCLUDE_DIRS 
        "${CMAKE_CURRENT_SOURCE_DIR}/hdf5/src"
        "${CMAKE_CURRENT_BINARY_DIR}/hdf5-build"
        PARENT_SCOPE
    )
    
    target_link_libraries(HDF5Wrapper PRIVATE
        hdf5-static
    )
    target_include_directories(HDF5Wrapper PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/hdf5/src
        ${CMAKE_CURRENT_BINARY_DIR}/hdf5-build
    )
    
else()
    # 默认：使用系统已安装的 HDF5
    message(STATUS "Using system-installed HDF5")
    
    find_package(HDF5 REQUIRED COMPONENTS C)
    
    # 设置 HDF5 库路径
    set(EZCOMPUTE_HDF5_LIBS "${HDF5_C_LIBRARIES}" PARENT_SCOPE)
    set(EZCOMPUTE_HDF5_INCLUDE_DIRS "${HDF5_C_INCLUDE_DIRS}" PARENT_SCOPE)
    
    target_link_libraries(HDF5Wrapper PRIVATE
        ${HDF5_C_LIBRARIES}
    )
    target_include_directories(HDF5Wrapper PRIVATE
        ${HDF5_C_INCLUDE_DIRS}
    )
    target_compile_definitions(HDF5Wrapper PRIVATE
        ${HDF5_C_DEFINITIONS}
    )
    
endif()
