################################################################################
# Mex Wrapper
################################################################################
add_subdirectory(wrappers)

################################################################################
# MATLAB API
################################################################################
set(SOURCE_FILES
        "arrus/Us4R.m"
        "arrus/Us4RSystem.m"
        "arrus/BModeDisplay.m"
        "arrus/Operation.m"
        "arrus/SimpleTxRxSequence.m"
        "arrus/PWISequence.m"
        "arrus/STASequence.m"
        "arrus/LINSequence.m"
        "arrus/mustBeDivisible.m"
        "arrus/probeParams.m"
        "arrus/downConversion.m"
        "arrus/reconstructRfImg.m"
        "arrus/reconstructRfLin.m"
        "arrus/scanConversion.m"
        "examples/Us4R_control.m"
        "examples/Us4R_maxSequence.m"
        "examples/Us4RUltrasonix_control.m"

        +arrus/MexObject.m
        +arrus/+session/Session.m
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

install(
        DIRECTORY
        ${TOOLBOX_OUTPUT_DIR}
        DESTINATION
        ${ARRUS_MATLAB_INSTALL_DIR}
)

install(
        FILES
        #TODO(pjarosik) to be consistent, move to CURRENT_BINARY_DIR
        ${CMAKE_CURRENT_SOURCE_DIR}/examples/Us4R_control.m
        ${CMAKE_CURRENT_SOURCE_DIR}/examples/Us4R_maxSequence.m
        ${CMAKE_CURRENT_SOURCE_DIR}/examples/Us4RUltrasonix_control.m
        DESTINATION
        ${ARRUS_MATLAB_INSTALL_DIR}/examples
)

# New API
# install(
#        DIRECTORY
#        ${CMAKE_CURRENT_SOURCE_DIR}/+arrus
#        DESTINATION
#        ${ARRUS_MATLAB_INSTALL_DIR}
#)

