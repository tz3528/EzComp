# --- EzCompile/Backend/Link/CMakeLists.txt ---

add_library(BackendLink STATIC
        Linker.cpp
)

target_include_directories(BackendLink PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

llvm_map_components_to_libnames(llvm_link_libs Support Core TargetParser)

target_link_libraries(BackendLink PUBLIC ${llvm_link_libs} MLIRSupport)

target_link_libraries(Backend PUBLIC BackendLink)