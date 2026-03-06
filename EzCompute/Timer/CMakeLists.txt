# --- EzCompute/Timing/CMakeLists.txt ---

add_library(Timer STATIC)

target_sources(Timer PRIVATE
        Timer.cpp
)

set(EZCOMPUTE_TIMER_REL "lib/libTimer.a" PARENT_SCOPE)
