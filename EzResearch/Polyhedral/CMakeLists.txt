# --- EzResearch/Polyhedral/CMakeLists.txt ---

target_sources(EzResearch PRIVATE
        AffineSystem.cpp
        DiamondTiling.cpp
        LoopSkewing.cpp
        OptPolyhedral.cpp
        PolyhedralInfo.cpp
)

# 查找并链接 Z3
find_package(Z3 REQUIRED PATHS "${CMAKE_CURRENT_SOURCE_DIR}/z3-install/lib/cmake/z3" NO_DEFAULT_PATH)
target_link_libraries(EzResearch PRIVATE z3::libz3)
target_include_directories(EzResearch PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/z3-install/include")
