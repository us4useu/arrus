# based on: https://github.com/Mizux/cmake-swig/blob/master/cmake/python.cmake
find_package(PythonInterp REQUIRED)

function(search_python_module MODULE_NAME)
    execute_process(
        COMMAND ${PYTHON_EXECUTABLE}
            -c "import ${MODULE_NAME}; print(${MODULE_NAME}.__version__)"
        RESULT_VARIABLE _RESULT
        OUTPUT_VARIABLE MODULE_VERSION
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(${_RESULT} STREQUAL "0")
        message(STATUS
            "Found python module: ${MODULE_NAME} (found version \"${MODULE_VERSION}\")")
    else()
        message(
            FATAL_ERROR
            "Can't find python module \"${MODULE_NAME}\", install it using pip install ${MODULE_NAME}")
    endif()
endfunction()

function(create_python_venv TARGET_NAME VENV_WORKING_DIR)
    search_python_module(virtualenv)
    set(VENV_EXECUTABLE ${PYTHON_EXECUTABLE} -m virtualenv)
    set(VENV_DIR "${VENV_WORKING_DIR}/venv")
    set(VENV_TIMESTAMP "${VENV_DIR}/timestamp")
    if(WIN32)
        set(VENV_BIN_DIR ${VENV_DIR}\\Scripts)
        set(VENV_PYTHON_EXECUTABLE "${VENV_BIN_DIR}\\python.exe")
    else()
        set(VENV_BIN_DIR ${VENV_DIR}/bin)
        set(VENV_PYTHON_EXECUTABLE ${VENV_BIN_DIR}/python)
    endif()
    add_custom_command(OUTPUT ${VENV_TIMESTAMP}
        COMMAND
            ${VENV_EXECUTABLE} -p ${PYTHON_EXECUTABLE} ${VENV_DIR}
        COMMAND
            ${CMAKE_COMMAND} -E touch ${VENV_TIMESTAMP}
        BYPRODUCTS
            ${VENV_DIR}
        WORKING_DIRECTORY
            ${VENV_WORKING_DIR}
    )
    add_custom_target(${TARGET_NAME} ALL DEPENDS ${VENV_TIMESTAMP})
    set_target_properties(${TARGET_NAME}
        PROPERTIES
            VENV_EXECUTABLE ${VENV_PYTHON_EXECUTABLE}
            VENV_DIR ${VENV_DIR}
            VENV_TIMESTAMP ${VENV_TIMESTAMP}
    )
endfunction()

function(install_arius_package TARGET_NAME VENV_TARGET PACKAGE_TARGET)
    get_target_property(INSTALL_VENV_EXECUTABLE ${VENV_TARGET} VENV_EXECUTABLE)
    get_target_property(INSTALL_VENV_DIR ${VENV_TARGET} VENV_DIR)
    get_target_property(ARIUS_PACKAGE_NAME ${PACKAGE_TARGET} PACKAGE_NAME)
    get_target_property(ARIUS_PACKAGE_DIR ${PACKAGE_TARGET} PACKAGE_DIR)
    get_target_property(ARIUS_PACKAGE_STAMP ${PACKAGE_TARGET} PACKAGE_TIMESTAMP)

    set(INSTALL_TIMESTAMP ${INSTALL_VENV_DIR}/${TARGET_NAME}_timestamp)

    add_custom_command(OUTPUT ${INSTALL_TIMESTAMP}
        COMMAND
            ${CMAKE_COMMAND} -E touch ${INSTALL_TIMESTAMP}
        COMMAND
            ${INSTALL_VENV_EXECUTABLE}
            -m pip install --upgrade --force-reinstall
            #TODO(pjarosik) consider appending timestamp to project version
            # in order to avoid unecessary reinstallation of arius dependencies
            --find-links=${ARIUS_PACKAGE_DIR} ${ARIUS_PACKAGE_NAME}
        DEPENDS
            ${VENV_TARGET} ${PACKAGE_TARGET} ${ARIUS_PACKAGE_STAMP}
        WORKING_DIRECTORY
            ${CURRENT_BINARY_DIR}
    )
    add_custom_target(${TARGET_NAME} ALL DEPENDS ${INSTALL_TIMESTAMP})
    set_target_properties(${TARGET_NAME}
        PROPERTIES
            INSTALL_TIMESTAMP ${INSTALL_TIMESTAMP}
    )
endfunction()

function(install_sphinx_package TARGET_NAME VENV_TARGET)
    get_target_property(INSTALL_VENV_EXECUTABLE ${VENV_TARGET} VENV_EXECUTABLE)
    get_target_property(INSTALL_VENV_DIR ${VENV_TARGET} VENV_DIR)
    set(INSTALL_TIMESTAMP ${INSTALL_VENV_DIR}/${TARGET_NAME}_timestamp)

    add_custom_command(OUTPUT ${INSTALL_TIMESTAMP}
        COMMAND
            ${CMAKE_COMMAND} -E touch ${INSTALL_TIMESTAMP}
        COMMAND
            ${INSTALL_VENV_EXECUTABLE} -m pip install sphinx sphinx_rtd_theme
            sphinxcontrib-matlabdomain
            #TODO(pjarosik) using here BUILD_MATLAB option to avoid unneceassry installation
        DEPENDS
            ${VENV_TARGET}
        WORKING_DIRECTORY
            ${CURRENT_BINARY_DIR}
    )
    add_custom_target(${TARGET_NAME} ALL DEPENDS ${INSTALL_TIMESTAMP})

    if(WIN32)
        set(VENV_SPHINX_EXECUTABLE "${INSTALL_VENV_DIR}\\Scripts\\sphinx-build.exe")
    else()
        set(VENV_SPHINX_EXECUTABLE ${INSTALL_VENV_DIR}/bin/sphinx-build)
    endif()

    set_target_properties(${TARGET_NAME}
        PROPERTIES
            INSTALL_TIMESTAMP ${INSTALL_TIMESTAMP}
            SPHINX_EXECUTABLE ${VENV_SPHINX_EXECUTABLE}
    )
endfunction()
