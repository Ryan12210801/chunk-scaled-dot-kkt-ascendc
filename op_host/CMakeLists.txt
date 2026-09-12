file(GLOB host_ops_def_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/*def.cpp
)

file(GLOB host_ops_infershape_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/*_infershape.cpp
)

set(host_ops_tiling_srcs)
file(GLOB TILING_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*tiling.cpp)
list(APPEND host_ops_tiling_srcs ${TILING_FILES})

set(host_ops_srcs
    ${host_ops_def_srcs}
    ${host_ops_infershape_srcs}
    ${host_ops_tiling_srcs}
)

npu_op_code_gen(
    SRC ${host_ops_srcs}
    PACKAGE ${package_name}
    OUT_DIR ${ASCEND_AUTOGEN_PATH}
    COMPILE_OPTIONS
        -I$ENV{ASCEND_HOME_PATH}/aarch64-linux/include
        -I$ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include/tiling
        -I$ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc
        -I$ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/op_common
        -I$ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/base
        -I$ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/exe_graph
        -I$ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/graph
)

npu_op_library(cust_optiling TILING
    ${host_ops_srcs}
)

target_include_directories(cust_optiling PRIVATE
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/include
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include/tiling
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/op_common
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/base
)

set(op_api_dir ${CMAKE_CURRENT_SOURCE_DIR}/../op_api)
if(EXISTS ${op_api_dir} AND IS_DIRECTORY ${op_api_dir})
    file(GLOB op_api_srcs ${op_api_dir}/*.cpp)
else()
    file(GLOB op_api_srcs "${CMAKE_BINARY_DIR}/autogen/aclnn_*.cpp")
endif()

npu_op_library(cust_opapi ACLNN
    ${op_api_srcs}
)

target_include_directories(cust_opapi PRIVATE
    ${op_api_dir}
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/include
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/include/aclnn
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/op_common
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/base
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/aicpu
)

target_compile_options(cust_opapi PRIVATE -UACLNN_WITH_BINARY)

file(GLOB proto_src ${ASCEND_AUTOGEN_PATH}/op_proto.cc)
set_source_files_properties(${proto_src} PROPERTIES GENERATED TRUE)

npu_op_library(cust_op_proto GRAPH
    ${host_ops_srcs}
    ${proto_src}
)

target_include_directories(cust_op_proto PRIVATE
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/include
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include/tiling
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/op_common
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/base
)

npu_op_package_add(${package_name}
    LIBRARY
        cust_optiling
        cust_op_proto
        cust_opapi
)

# Install hand-written op_api headers into RUN package
if(EXISTS ${op_api_dir} AND IS_DIRECTORY ${op_api_dir})
    file(GLOB op_api_aclnn_headers ${op_api_dir}/aclnn_*.h ${op_api_dir}/acl_*.h)
    if(op_api_aclnn_headers)
        npu_op_package_add(${package_name}
            FILES ${op_api_aclnn_headers}
            TYPE ACLNN
        )
    endif()
endif()
