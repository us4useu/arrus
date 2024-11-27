################################################################################
# Mex Wrapper
################################################################################
add_subdirectory(wrappers)

################################################################################
# MATLAB API
################################################################################
set(SOURCE_FILES
    arrus/BModeDisplay.m
    arrus/CustomTxRxSequence.m
    arrus/dopplerColorImaging.m
    arrus/downConversion.m
    arrus/DuplexDisplay.m
    arrus/mustBeDivisible.m
    arrus/Reconstruction.m
    arrus/RemapToLogicalOrder.m
    arrus/scanConversion.m
    arrus/Us4R.m
    arrus/Us4RHandle.m
    arrus/wedgeParams.m
    examples/bmodePwi.m
    examples/bmodeDwi.m
    examples/bmodeLin.m
    examples/colorPwi.m
    examples/rawRfPwi.m
    examples/rawRfPwiCustomSequence.m
)

set(ARRUS_CUDA_MEX_SRC_FILES
    arrus/mexcuda/rawReorg.cu
    arrus/mexcuda/wcFilter.cu
    arrus/mexcuda/dopplerColor.cu
    arrus/mexcuda/iqRaw2Lin.cu
    arrus/mexcuda/iqRaw2Lri.cu
    arrus/mexcuda/iqRaw2Lri_SSTA_Wedge.cu
)

################################################################################
# MATLAB TOOLBOX target
################################################################################
set(TIMESTAMP "${CMAKE_CURRENT_BINARY_DIR}/timestamp")

set(TOOLBOX_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/arrus)
set(PACKAGE_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/+arrus)

if("${ARRUS_BUILD_PLATFORM}" STREQUAL "windows")
# Note: the below mexcuda is supported only on Windows.
# For linux, use arrus framework kernels implementation.
# Generate mexcuda matlab command

# TODO remove it when mexcuda is not needed anymore
set(ARRUS_MATLAB_MEXCUDA_COMMAND "")
foreach(MEX_SRC_FILE ${ARRUS_CUDA_MEX_SRC_FILES})
    set(ARRUS_MATLAB_MEXCUDA_COMMAND
        "${ARRUS_MATLAB_MEXCUDA_COMMAND} mexcuda -outdir '${CMAKE_CURRENT_BINARY_DIR}/arrus/mexcuda' '${CMAKE_CURRENT_SOURCE_DIR}/${MEX_SRC_FILE}';")
endforeach()

set(ARRUS_MATLAB_CORE_SO_FILE "$<TARGET_FILE:arrus-core>")

add_custom_command(OUTPUT ${TIMESTAMP}
    COMMAND
    ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/arrus ${TOOLBOX_OUTPUT_DIR}
    COMMAND
    ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/+arrus ${PACKAGE_OUTPUT_DIR}
    COMMAND
    ${CMAKE_COMMAND} -E copy_directory
    ${Us4_ROOT_DIR}/lib64 ${TOOLBOX_OUTPUT_DIR}
    COMMAND
    ${CMAKE_COMMAND} -E copy
    ${ARRUS_MATLAB_CORE_SO_FILE} ${TOOLBOX_OUTPUT_DIR}
    # Excluded directories/files
    COMMAND
    ${CMAKE_COMMAND} -E rm -rf "${TOOLBOX_OUTPUT_DIR}/mexcuda"
    # Generate .mex files for image reconstruction
    COMMAND
    ${CMAKE_COMMAND} -E make_directory "${TOOLBOX_OUTPUT_DIR}/mexcuda"
    COMMAND
    matlab -batch "${ARRUS_MATLAB_MEXCUDA_COMMAND}"
    COMMAND
    ${CMAKE_COMMAND} -E touch ${TIMESTAMP}
    DEPENDS ${SOURCE_FILES}
)
else()
    add_custom_command(OUTPUT ${TIMESTAMP}
        COMMAND
        ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/arrus ${TOOLBOX_OUTPUT_DIR}
        COMMAND
        ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/+arrus ${PACKAGE_OUTPUT_DIR}
        COMMAND
        ${CMAKE_COMMAND} -E copy_directory
        ${Us4_ROOT_DIR}/lib64 ${TOOLBOX_OUTPUT_DIR}
        COMMAND
        ${CMAKE_COMMAND} -E touch ${TIMESTAMP}
        DEPENDS ${SOURCE_FILES}
        )
endif()


add_custom_target(matlab_toolbox ALL DEPENDS ${TIMESTAMP})

set_target_properties(
    matlab_toolbox
    PROPERTIES
    ARRUS_TIMESTAMP ${TIMESTAMP}
    MATLAB_TOOLBOX_BIN_DIR ${CMAKE_CURRENT_BINARY_DIR}
)

# The old Matlab files.
install(
    DIRECTORY
    ${TOOLBOX_OUTPUT_DIR}
    DESTINATION
    ${ARRUS_MATLAB_INSTALL_DIR}
)
install(
    DIRECTORY
    ${PACKAGE_OUTPUT_DIR}
    DESTINATION
    ${ARRUS_MATLAB_INSTALL_DIR}
)

install(
    DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR}/examples
    DESTINATION
    ${ARRUS_MATLAB_INSTALL_DIR}
)
