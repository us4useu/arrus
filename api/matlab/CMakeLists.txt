################################################################################
# MATLAB API
################################################################################
set(SOURCE_FILES
    "arrus/Us4R.m"
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
add_custom_command(OUTPUT ${TIMESTAMP}
    COMMAND
        ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/arrus ${CMAKE_CURRENT_BINARY_DIR}/arrus
    COMMAND
        ${CMAKE_COMMAND} -E touch ${TIMESTAMP}
    DEPENDS ${SOURCE_FILES}
)

add_custom_target(matlab_toolbox ALL DEPENDS ${TIMESTAMP})
