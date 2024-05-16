include(CMakeFindDependencyMacro)
find_dependency(OpenVINO COMPONENTS Runtime)

if(NOT TARGET genai)
    include("${CMAKE_CURRENT_LIST_DIR}/openvino_genaiTargets.cmake")
endif()
