cmake_minimum_required(VERSION 3.14)
project(ACLNN_EXAMPLE)

add_compile_options(-std=c++17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "./bin")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

add_executable(test_aclnn_chunk_scaled_dot_kkt
test_aclnn_chunk_scaled_dot_kkt.cpp)

if(NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
    set(ASCEND_PATH $ENV{ASCEND_HOME_PATH})
else()
    set(ASCEND_PATH "/usr/local/Ascend/cann")
endif()

find_path(CUSTOM_OP_INCLUDE_DIR
    NAMES aclnn_chunk_scaled_dot_kkt.h
    PATHS
        ${ASCEND_PATH}/opp/vendors/chunk_scaled_dot_kkt_custom/op_api/include
        /usr/local/Ascend/opp/vendors/chunk_scaled_dot_kkt_custom/op_api/include
        $ENV{HOME}/Ascend/opp/vendors/chunk_scaled_dot_kkt_custom/op_api/include
)

if(NOT CUSTOM_OP_INCLUDE_DIR)
    message(FATAL_ERROR "未找到自定义算子头文件 aclnn_chunk_scaled_dot_kkt.h，请先安装算子包")
endif()
message(STATUS "自定义算子头文件目录: ${CUSTOM_OP_INCLUDE_DIR}")

find_library(CUSTOM_OP_LIBRARY cust_opapi
    PATHS
        ${ASCEND_PATH}/opp/vendors/chunk_scaled_dot_kkt_custom/op_api/lib
        /usr/local/Ascend/opp/vendors/chunk_scaled_dot_kkt_custom/op_api/lib
        $ENV{HOME}/Ascend/opp/vendors/chunk_scaled_dot_kkt_custom/op_api/lib
)

if(NOT CUSTOM_OP_LIBRARY)
    message(FATAL_ERROR "未找到自定义算子库 libcust_opapi.so，请先安装算子包")
endif()

include_directories(
    ${ASCEND_PATH}/include
    ${CUSTOM_OP_INCLUDE_DIR}
)

target_link_libraries(test_aclnn_chunk_scaled_dot_kkt PRIVATE
    ${CUSTOM_OP_LIBRARY}
    ${ASCEND_PATH}/lib64/libascendcl.so
    ${ASCEND_PATH}/lib64/libnnopbase.so
    ${ASCEND_PATH}/lib64/libopapi.so
)
get_filename_component(CUSTOM_OP_LIB_DIR ${CUSTOM_OP_LIBRARY} DIRECTORY)
target_link_options(test_aclnn_chunk_scaled_dot_kkt PRIVATE
    "-Wl,-rpath,${CUSTOM_OP_LIB_DIR}"
)

install(TARGETS test_aclnn_chunk_scaled_dot_kkt DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
