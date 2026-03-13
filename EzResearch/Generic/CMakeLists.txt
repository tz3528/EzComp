# --- EzResearch/Generic/CMakeLists.txt ---

target_sources(EzResearch PRIVATE
    OptLoopPeeling.cpp
)

# 添加子模块
add_subdirectory(BoundaryHoist)
