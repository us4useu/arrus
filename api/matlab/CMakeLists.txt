################################################################################
# MATLAB API
################################################################################
set(SOURCE_FILES
    "Us4R.m"
    "probeParams.m"
)

set(TIMESTAMP "${CMAKE_CURRENT_BINARY_DIR}/timestamp")

add_custom_command(OUTPUT ${TIMESTAMP}
    COMMAND
        ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/arrus ${CMAKE_CURRENT_BINARY_DIR}/arrus
    COMMAND
        ${CMAKE_COMMAND} -E touch ${TIMESTAMP}
    COMMAND
        ${PYTHON_EXECUTABLE} ${SETUP_PY_OUT} bdist_wheel
    DEPENDS ${SETUP_PY_IN} ${SOURCE_FILES} ${Us4Wrappers}
)

add_custom_target(python_whl ALL DEPENDS ${TIMESTAMP})
set_target_properties(
    python_whl
    PROPERTIES
        PACKAGE_TIMESTAMP ${TIMESTAMP}
        PACKAGE_NAME ${PYTHON_PACKAGE_NAME}
        PACKAGE_DIR ${CMAKE_CURRENT_BINARY_DIR}/dist
)
