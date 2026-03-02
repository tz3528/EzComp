# --- EzCompute/IO/CMakeLists.txt ---

add_library(HDF5 STATIC)

target_sources(HDF5 PRIVATE
        HDF5.cpp
)

set(EZCOMPUTE_HDF5_REL  "lib/libHDF5.a" PARENT_SCOPE)

find_package(HDF5 REQUIRED COMPONENTS C)

target_include_directories(HDF5 PRIVATE ${HDF5_INCLUDE_DIRS})

target_link_libraries(HDF5 PRIVATE ${HDF5_LIBRARIES})
