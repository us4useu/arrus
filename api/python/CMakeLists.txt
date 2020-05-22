################################################################################
# Python Swig wrappers
################################################################################
# - Targets
################################################################################

if(ARRUS_BUILD_SWIG)
find_package(SWIG REQUIRED)
include(UseSWIG)
find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
set(Us4Components
    US4OEM
    HV256
    DBARLite
)
set(Us4Wrappers
    ius4oem
    ihv256
    idbarlite
)

foreach(wrapper component IN ZIP_LISTS Us4Wrappers Us4Components)
    set_property(SOURCE wrappers/${wrapper}.i PROPERTY CPLUSPLUS ON)
    if(MSVC)
        set_property(
            SOURCE wrappers/${wrapper}.i
            PROPERTY
                GENERATED_COMPILE_OPTIONS /Od /EHsc
        )
    endif()
    swig_add_library(${wrapper}
        TYPE SHARED
        LANGUAGE PYTHON
        OUTPUT_DIR arrus/devices
        OUTFILE_DIR wrappers
        SOURCES wrappers/${wrapper}.i
    )
    set_target_properties(${wrapper}
        PROPERTIES
            SWIG_USE_TARGET_INCLUDE_DIRECTORIES ON
            RUNTIME_OUTPUT_DIRECTORY
            "${CMAKE_CURRENT_BINARY_DIR}/arrus/devices/$<$<CONFIG:Debug>:>"
    )
    target_include_directories(${wrapper}
        PRIVATE
        ${PYTHON_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_link_libraries(${wrapper}
       PRIVATE
       core
       Us4::${component}
        ${PYTHON_LIBRARIES}
    )
endforeach()

endif()
################################################################################
# Python packaging
################################################################################
include(python)
set(PYTHON_PACKAGE_NAME arrus)

set(SOURCE_FILES
    "arrus/__init__.py"
    "arrus/beam.py"
    "arrus/interface.py"
    "arrus/session.py"
    "arrus/kernels.py"
    "arrus/operations.py"
    "arrus/params.py"
    "arrus/validation.py"
    "arrus/devices/__init__.py"
    "arrus/devices/us4oem.py"
    "arrus/devices/device.py"
    "arrus/devices/hv256.py"
    "arrus/devices/probe.py"
    "arrus/utils/__init__.py"
    "arrus/utils/imaging.py"
    "arrus/utils/parameters.py"
    "arrus/devices/callbacks.py"
    "arrus/tests/tools.py"
)
set(TEST_FILES
    "arrus/tests/beam_test.py"
    "arrus/tests/session_test.py"
    "arrus/devices/tests/probe_test.py"
    "arrus/utils/tests/utils_test.py"
    "arrus/utils/tests/parameters_test.py"
    "arrus/utils/tests/us4oem_kernels_test.py"
)

set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
set(SETUP_PY_OUT "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
set(TIMESTAMP "${CMAKE_CURRENT_BINARY_DIR}/timestamp")

configure_file(${SETUP_PY_IN} ${SETUP_PY_OUT})

add_custom_command(OUTPUT ${TIMESTAMP}
    COMMAND
        ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/arrus ${CMAKE_CURRENT_BINARY_DIR}/arrus
    COMMAND
        ${CMAKE_COMMAND} -E touch ${TIMESTAMP}
    COMMAND
        ${PYTHON_EXECUTABLE} ${SETUP_PY_OUT} bdist_wheel
    DEPENDS ${SETUP_PY_IN} ${SOURCE_FILES} ${TEST_FILES} ${Us4Wrappers}
)

add_custom_target(python_whl ALL DEPENDS ${TIMESTAMP})
set_target_properties(
    python_whl
    PROPERTIES
        PACKAGE_TIMESTAMP ${TIMESTAMP}
        PACKAGE_NAME ${PYTHON_PACKAGE_NAME}
        PACKAGE_SETUP_PY_DIR "${CMAKE_CURRENT_BINARY_DIR}"
        PACKAGE_DIR ${CMAKE_CURRENT_BINARY_DIR}/dist
)

################################################################################
# Copy .whl package to install directory
################################################################################
get_target_property(ARRUS_WHL_DIR python_whl PACKAGE_DIR)

install(
    FILES
        "${ARRUS_WHL_DIR}/${PYTHON_PACKAGE_NAME}-${PROJECT_VERSION}-py3-none-any.whl"
    DESTINATION
        ${ARRUS_PYTHON_INSTALL_DIR}
    RENAME
        #TODO(pjarosik) make it more general and available for other OSes
        "${PYTHON_PACKAGE_NAME}-${PROJECT_VERSION}-cp37-cp37m-win_amd64.whl"
    COMPONENT
        python_whl
)

# install(
#	FILES
#        # TODO (pjarosik) to be consistent, move to CURRENT_BUILD_DIR
#	    ${CMAKE_CURRENT_SOURCE_DIR}/examples/
#    DESTINATION
#        ${ARRUS_PYTHON_INSTALL_DIR}/examples/
#)

################################################################################
# Python tests
################################################################################
if(ARRUS_RUN_TESTS)
    create_python_venv(py_tests_venv ${CMAKE_CURRENT_BINARY_DIR})
    install_arrus_package(py_tests_venv_arius py_tests_venv python_whl "DIR")
    get_target_property(PY_TESTS_VENV_EXEC py_tests_venv VENV_EXECUTABLE)
    foreach(TEST_FILE IN LISTS TEST_FILES)
        add_test(
            NAME ${TEST_FILE}
            COMMAND ${PY_TESTS_VENV_EXEC} -m unittest ${CMAKE_CURRENT_BINARY_DIR}/${TEST_FILE}
        )
    endforeach()
endif()

