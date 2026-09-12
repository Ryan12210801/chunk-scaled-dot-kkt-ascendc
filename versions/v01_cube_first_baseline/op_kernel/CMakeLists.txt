file(GLOB_RECURSE ALL_KERNEL_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

npu_op_kernel_sources(ascendc_kernels
    OP_TYPE OP
    KERNEL_DIR .
    KERNEL_FILE ${ALL_KERNEL_FILES}
)

npu_op_kernel_library(ascendc_kernels
    SRC_BASE ${CMAKE_CURRENT_SOURCE_DIR}
    TILING_LIBRARY cust_optiling
)

npu_op_package_add(${package_name}
    LIBRARY
        ascendc_kernels
)
