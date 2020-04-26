################################################################################
# MATLAB API
################################################################################
set(SOURCE_FILES
    "arrus/Us4R.m"
    "arrus/Us4RDisplay.m"
    "arrus/Operation.m"
    "arrus/SimpleTxRxSequence.m"
    "arrus/PWISequence.m"
    "arrus/STASequence.m"
    "arrus/LINSequence.m"
    "arrus/mustBeDivisible.m"
    "arrus/probeParams.m"
    "arrus/downConv.m"
    "arrus/reconstructRfImg.m"
    "arrus/reconstructRfLin.m"
    "arrus/scanConv.m"
)

################################################################################
# MATLAB TOOLBOX target
################################################################################
set(TIMESTAMP "${CMAKE_CURRENT_BINARY_DIR}/timestamp")

set(TOOLBOX_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/arrus)

add_custom_command(OUTPUT ${TIMESTAMP}
    COMMAND
        ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/arrus ${TOOLBOX_OUTPUT_DIR}
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


