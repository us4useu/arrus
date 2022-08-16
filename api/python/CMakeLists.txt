################################################################################
# Python Swig wrappers
################################################################################
# - Targets
################################################################################
find_package(SWIG REQUIRED)
include(UseSWIG)
find_package(PythonInterp ${ARRUS_PY_VERSION} REQUIRED)
find_package(PythonLibs ${ARRUS_PY_VERSION} REQUIRED)
set_property(SOURCE wrappers/core.i PROPERTY CPLUSPLUS ON)
if (MSVC)
    set_property(
            SOURCE wrappers/core.i
            PROPERTY
            GENERATED_COMPILE_OPTIONS /Od /EHsc
    )
endif ()
swig_add_library(py_core
        TYPE SHARED
        LANGUAGE PYTHON
        OUTPUT_DIR arrus
        OUTFILE_DIR wrappers
        SOURCES
        wrappers/core.i)

if ("${ARRUS_BUILD_PLATFORM}" STREQUAL "windows")
    set_target_properties(py_core
            PROPERTIES
            SWIG_USE_TARGET_INCLUDE_DIRECTORIES ON
            RUNTIME_OUTPUT_DIRECTORY
            "${CMAKE_CURRENT_BINARY_DIR}/arrus/$<$<CONFIG:Debug>:>"
            )
elseif ("${ARRUS_BUILD_PLATFORM}" STREQUAL "linux")
    set_target_properties(py_core
            PROPERTIES
            SWIG_USE_TARGET_INCLUDE_DIRECTORIES ON
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/arrus/"
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/arrus/"
            ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/arrus/"
            # Skip build rpath, RPATH will be set to $ORIGIN using patchelf
            # when preparing whl package
            SKIP_BUILD_RPATH ON
            )
endif ()

target_include_directories(py_core
        PRIVATE
        ${PYTHON_INCLUDE_DIRS}
        ${ARRUS_ROOT_DIR}
        )

target_link_libraries(py_core
        PRIVATE
        arrus-core
        Boost::Boost
        fmt::fmt
        ${PYTHON_LIBRARIES}
        )
################################################################################
# Python packaging
################################################################################
include(python)
set(PYTHON_PACKAGE_NAME arrus)

set(SOURCE_FILES
        arrus/__init__.py
        arrus/logging.py
        arrus/session.py
        arrus/validation.py
        arrus/metadata.py
        arrus/exceptions.py
        arrus/medium.py
        arrus/params.py
        arrus/framework.py
        arrus/ops/__init__.py
        arrus/ops/imaging.py
        arrus/ops/tgc.py
        arrus/ops/operation.py
        arrus/ops/us4r.py
        arrus/devices/__init__.py
        arrus/devices/cpu.py
        arrus/devices/gpu.py
        arrus/devices/us4r.py
        arrus/devices/probe.py
        arrus/devices/device.py
        arrus/kernels/__init__.py
        arrus/kernels/kernel.py
        arrus/kernels/imaging.py
        arrus/kernels/tgc.py
        arrus/utils/__init__.py
        arrus/utils/imaging.py
        arrus/utils/us4r.py
        arrus/utils/us4r_remap_gpu.py
        arrus/utils/fir.py
        arrus/utils/interpolate.py
        arrus/utils/core.py
        arrus/utils/gui.py
        arrus/utils/probe_check.py
        )

set(TEST_FILES
#    arrus/kernels/tests/imaging_test.py
)

if(ARRUS_CUDA)
    # CUDA GPU specific tests.
    # CuPy is required.
    set(TEST_FILES
        arrus/utils/tests/pipeline_test.py
        ${TEST_FILES})
endif()


set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
set(SETUP_PY_OUT "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
set(INIT_PY_IN  "${CMAKE_CURRENT_SOURCE_DIR}/arrus/__init__.py")
# The below output will be moved to the arrus subdirectory (by custom_command),
# right after the .py source files are available in build_dir.
set(INIT_PY_OUT "${CMAKE_CURRENT_BINARY_DIR}/tmp/__init__.py")
set(TIMESTAMP "${CMAKE_CURRENT_BINARY_DIR}/timestamp")

configure_file(${SETUP_PY_IN} ${SETUP_PY_OUT})
configure_file(${INIT_PY_IN}  ${INIT_PY_OUT})


if ("${ARRUS_BUILD_PLATFORM}" STREQUAL "windows")
    set(ARRUS_TARGET_SHARED_OBJECT_FILE "${PROJECT_BINARY_DIR}/arrus/core/Release/arrus-core.dll")

    add_custom_command(OUTPUT ${TIMESTAMP}
            COMMAND
            ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/arrus ${CMAKE_CURRENT_BINARY_DIR}/arrus
            # Override __init__.py file with the correct Python package version.
            COMMAND
            ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_BINARY_DIR}/tmp/__init__.py ${CMAKE_CURRENT_BINARY_DIR}/arrus
            COMMAND
            ${CMAKE_COMMAND} -E copy_directory
            ${Us4_ROOT_DIR}/lib64 ${CMAKE_CURRENT_BINARY_DIR}/arrus
            COMMAND
            ${CMAKE_COMMAND} -E copy
            ${ARRUS_TARGET_SHARED_OBJECT_FILE} ${CMAKE_CURRENT_BINARY_DIR}/arrus
            COMMAND
            ${CMAKE_COMMAND} -E touch ${TIMESTAMP}
            COMMAND
            ${PYTHON_EXECUTABLE} ${SETUP_PY_OUT} bdist_wheel
            DEPENDS ${SETUP_PY_IN} ${SOURCE_FILES} ${TEST_FILES} py_core)

elseif ("${ARRUS_BUILD_PLATFORM}" STREQUAL "linux")
    set(ARRUS_CORE_SO_FILE "$<TARGET_FILE:arrus-core>")
    set(ARRUS_CORE_SO_FILE_NAME "$<TARGET_FILE_NAME:arrus-core>")
    set(PY_CORE_SO_FILE_NAME "$<TARGET_FILE_NAME:py_core>")
    add_custom_command(OUTPUT ${TIMESTAMP}
            COMMAND
            ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/arrus ${CMAKE_CURRENT_BINARY_DIR}/arrus
            # Override __init__.py file with the correct Python package version.
            COMMAND
            ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_BINARY_DIR}/tmp/__init__.py ${CMAKE_CURRENT_BINARY_DIR}/arrus
            COMMAND
            ${CMAKE_COMMAND} -E copy_directory
            ${Us4_ROOT_DIR}/lib64 ${CMAKE_CURRENT_BINARY_DIR}/arrus
            # Assumming RPATH is already set to $ORIGIN
            COMMAND
            ${CMAKE_COMMAND} -E copy
            ${ARRUS_CORE_SO_FILE} ${CMAKE_CURRENT_BINARY_DIR}/arrus
            COMMAND
            patchelf --set-rpath '$$ORIGIN' ${CMAKE_CURRENT_BINARY_DIR}/arrus/${ARRUS_CORE_SO_FILE_NAME}
            COMMAND
            patchelf --set-rpath '$$ORIGIN' ${CMAKE_CURRENT_BINARY_DIR}/arrus/${PY_CORE_SO_FILE_NAME}
            COMMAND
            ${CMAKE_COMMAND} -E touch ${TIMESTAMP}
            COMMAND
            ${PYTHON_EXECUTABLE} ${SETUP_PY_OUT} bdist_wheel
            DEPENDS ${SETUP_PY_IN} ${SOURCE_FILES} ${TEST_FILES} py_core)
endif ()


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
        DIRECTORY 
           ${ARRUS_WHL_DIR}/
        DESTINATION
           ${ARRUS_PYTHON_INSTALL_DIR}
        COMPONENT
        python_whl
	FILES_MATCHING PATTERN "${PYTHON_PACKAGE_NAME}-*.whl"
)

install(
    FILES
        examples/plane_wave_imaging.py
        examples/classical_beamforming.py
        examples/diverging_beams.py
        examples/custom_tx_rx_sequence.py
        examples/custom_callback.py
	    examples/phased_array_scanning.py
        examples/check_probe.py
        examples/requirements.txt
        examples/requirements.txt
    DESTINATION
        ${ARRUS_PYTHON_INSTALL_DIR}/examples/
)
################################################################################
# Tests
################################################################################
if(ARRUS_RUN_TESTS)
    create_python_venv(py_tests_venv ${CMAKE_CURRENT_BINARY_DIR})
    install_arrus_package(py_tests_venv_arius py_tests_venv python_whl "DIR")
    get_target_property(PY_TESTS_VENV_EXEC py_tests_venv VENV_EXECUTABLE)
    foreach(TEST_FILE IN LISTS TEST_FILES)
        add_test(
            NAME ${TEST_FILE}
            COMMAND ${PY_TESTS_VENV_EXEC} -m unittest ${CMAKE_CURRENT_BINARY_DIR}/${TEST_FILE})
    endforeach()
endif()
