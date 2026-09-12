cmake_minimum_required(VERSION 3.16.0)
project(chunk_scaled_dot_kkt_op_prj)
find_package(ASC REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ASCEND_COMPUTE_UNIT OR ASCEND_COMPUTE_UNIT STREQUAL "")
    set(ASCEND_COMPUTE_UNIT ascend910b)
endif()

# The kernel uses Ascend 910B (NPU arch 2201) specific instructions and guards.
# Fail closed instead of implying that unverified SoCs are supported.
if(NOT ASCEND_COMPUTE_UNIT STREQUAL "ascend910b")
    message(FATAL_ERROR
        "This release has only been validated for ASCEND_COMPUTE_UNIT=ascend910b; "
        "got '${ASCEND_COMPUTE_UNIT}'.")
endif()
set(package_name chunk_scaled_dot_kkt_custom)

 npu_op_package(${package_name}
    TYPE RUN
    CONFIG
        INSTALL_PATH ${CMAKE_BINARY_DIR}
)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/op_host")
    add_subdirectory(op_host)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/op_kernel")
    add_subdirectory(op_kernel)
endif()

message(STATUS "Use 'bash build.sh' to build the kernel binary and install package.")
