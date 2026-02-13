# --- EzCompile/Midend/Conversions/CMakeLists.txt ---

target_sources(Midend PRIVATE
        LowerCompApplyInit.cpp
        LowerCompDim.cpp
        LowerCompDirichlet.cpp
        LowerCompField.cpp
        LowerCompForTime.cpp
        LowerCompPoints.cpp
        LowerCompProblem.cpp
        LowerCompSolve.cpp
        LowerCompUpdate.cpp
)