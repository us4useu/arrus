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
set_property(SOURCE wrappers/core.i PROPERTY CPLUSPLUS ON)
if(MSVC)
    set_property(
        SOURCE wrappers/core.i
        PROPERTY
        GENERATED_COMPILE_OPTIONS /Od /EHsc
    )
endif()
swig_add_library(py_core
    TYPE SHARED
    LANGUAGE PYTHON
    OUTPUT_DIR arrus
    OUTFILE_DIR wrappers
    SOURCES
		wrappers/core.i
		${ARRUS_ROOT_DIR}/arrus/common/logging/impl/Logging.cpp
		${ARRUS_ROOT_DIR}/arrus/common/logging/impl/LogSeverity.cpp
)

if("${ARRUS_BUILD_PLATFORM}" STREQUAL "windows")
set_target_properties(py_core
	PROPERTIES
		SWIG_USE_TARGET_INCLUDE_DIRECTORIES ON
		RUNTIME_OUTPUT_DIRECTORY
		"${CMAKE_CURRENT_BINARY_DIR}/arrus/$<$<CONFIG:Debug>:>"
)
elseif("${ARRUS_BUILD_PLATFORM}" STREQUAL "linux")
	set_target_properties(py_core
		PROPERTIES
		SWIG_USE_TARGET_INCLUDE_DIRECTORIES ON
		RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/arrus/"
		LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/arrus/"
		ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/arrus/"
		)
endif()

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

endif()
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
    arrus/utils/parameters.py
	arrus/utils/us4r.py
	arrus/utils/us4r_remap_gpu.py
	arrus/utils/fir.py
	arrus/utils/interpolate.py
	arrus/utils/core.py
	arrus/utils/gui.py
	arrus/utils/sta.cu
)

set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
set(SETUP_PY_OUT "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
set(TIMESTAMP "${CMAKE_CURRENT_BINARY_DIR}/timestamp")

configure_file(${SETUP_PY_IN} ${SETUP_PY_OUT})


if("${ARRUS_BUILD_PLATFORM}" STREQUAL "windows")
	set(ARRUS_TARGET_SHARED_OBJECT_FILE "${PROJECT_BINARY_DIR}/arrus/core/Release/arrus-core.dll")

    add_custom_command(OUTPUT ${TIMESTAMP}
    COMMAND
        ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/arrus ${CMAKE_CURRENT_BINARY_DIR}/arrus
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

elseif("${ARRUS_BUILD_PLATFORM}" STREQUAL "linux")
	set(ARRUS_TARGET_SHARED_OBJECT_FILE "$<TARGET_FILE:arrus-core>")
    add_custom_command(OUTPUT ${TIMESTAMP}
      COMMAND
        ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/arrus ${CMAKE_CURRENT_BINARY_DIR}/arrus
      COMMAND
        ${CMAKE_COMMAND} -E touch ${TIMESTAMP}
      COMMAND
        ${PYTHON_EXECUTABLE} ${SETUP_PY_OUT} bdist_wheel
      DEPENDS ${SETUP_PY_IN} ${SOURCE_FILES} ${TEST_FILES} py_core)
endif()



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
        "${PYTHON_PACKAGE_NAME}-${PROJECT_VERSION}-py3-none-any.whl"
    COMPONENT
        python_whl
)

install(
	FILES
	examples/sta_sequence_example.py
	examples/pwi_sequence_example.py
	examples/lin_sequence_example.py
	examples/custom_sequence_example.py
	examples/custom_callback_example.py
	examples/requirements.txt
	examples/gpu_requirements.txt
    DESTINATION
        ${ARRUS_PYTHON_INSTALL_DIR}/examples/
)

