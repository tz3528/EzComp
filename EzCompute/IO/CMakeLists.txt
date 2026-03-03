# --- EzCompute/IO/CMakeLists.txt ---

add_library(HDF5 STATIC)

target_sources(HDF5 PRIVATE
        HDF5.cpp
)

set(EZCOMPUTE_HDF5_REL  "lib/libHDF5.a" PARENT_SCOPE)

find_package(HDF5 REQUIRED COMPONENTS C)

target_include_directories(HDF5 PRIVATE ${HDF5_INCLUDE_DIRS})

target_link_libraries(HDF5 PRIVATE ${HDF5_LIBRARIES})

# 导出 HDF5 库路径供 manifest 使用（格式化为 JSON 数组）
# HDF5_LIBRARIES 是分号分隔的列表，转换为 "path1", "path2" 格式
set(_json_libs "")
foreach(_lib ${HDF5_LIBRARIES})
    if(_json_libs)
        string(APPEND _json_libs ", ")
    endif()
    string(APPEND _json_libs "\"${_lib}\"")
endforeach()
set(EZCOMPUTE_HDF5_LIBS "${_json_libs}" PARENT_SCOPE)
