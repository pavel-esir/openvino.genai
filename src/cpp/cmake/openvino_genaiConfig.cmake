include(CMakeFindDependencyMacro)
find_dependency(OpenVINO COMPONENTS Runtime)

include("${CMAKE_CURRENT_LIST_DIR}/openvino_genaiTargets.cmake")
