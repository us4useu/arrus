################################################################################
# Mex Wrapper
################################################################################
add_subdirectory(wrappers)

################################################################################
# MATLAB API
################################################################################
set(SOURCE_FILES
    arrus/BModeDisplay.m
    arrus/dopplerColorImaging.m
    arrus/downConversion.m
    arrus/DuplexDisplay.m
    arrus/LINSequence.m
    arrus/mustBeDivisible.m
    arrus/Operation.m
    arrus/probeParams.m
    arrus/PWISequence.m
    arrus/reconstructBModeImageLIN.m
    arrus/Reconstruction.m
    arrus/reconstructRfImg.m
    arrus/reconstructRfLin.m
    arrus/scanConversion.m
    arrus/SimpleTxRxSequence.m
    arrus/STASequence.m
    arrus/Us4R.m
    arrus/Us4RSystem.m

    examples/Us4R_control.m
    examples/Us4R_duplex.m
    examples/Us4R_maxSequence.m
    examples/Us4R_Olympus_control.m
    examples/Us4R_Vermon_control.m
    examples/Us4R_ATL_control.m
    examples/Us4R_Ultrasonix_control.m
)

set(ARRUS_CUDA_MEX_SRC_FILES
    arrus/mexcuda/dopplerColor.cu
    arrus/mexcuda/iqRaw2Lri.cu
)

################################################################################
# MATLAB TOOLBOX target
################################################################################
set(TIMESTAMP "${CMAKE_CURRENT_BINARY_DIR}/timestamp")

set(TOOLBOX_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/arrus)

# Generate mexcuda matlab command
set(ARRUS_MATLAB_MEXCUDA_COMMAND "")
foreach(MEX_SRC_FILE ${ARRUS_CUDA_MEX_SRC_FILES})
    set(ARRUS_MATLAB_MEXCUDA_COMMAND
        "${ARRUS_MATLAB_MEXCUDA_COMMAND} mexcuda -outdir '${CMAKE_CURRENT_BINARY_DIR}/arrus/mexcuda' '${CMAKE_CURRENT_SOURCE_DIR}/${MEX_SRC_FILE}';")
endforeach()


add_custom_command(OUTPUT ${TIMESTAMP}
    COMMAND
    ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/arrus ${TOOLBOX_OUTPUT_DIR}
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


add_custom_target(matlab_toolbox ALL DEPENDS ${TIMESTAMP})

set_target_properties(
    matlab_toolbox
    PROPERTIES
    ARRUS_TIMESTAMP ${TIMESTAMP}
    MATLAB_TOOLBOX_BIN_DIR ${CMAKE_CURRENT_BINARY_DIR}
)

install(
    DIRECTORY
    ${TOOLBOX_OUTPUT_DIR}
    DESTINATION
    ${ARRUS_MATLAB_INSTALL_DIR}
)

install(
    DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR}/examples
    DESTINATION
    ${ARRUS_MATLAB_INSTALL_DIR}
)

# New API
# install(
#        DIRECTORY
#        ${CMAKE_CURRENT_SOURCE_DIR}/+arrus
#        DESTINATION
#        ${ARRUS_MATLAB_INSTALL_DIR}
#)

