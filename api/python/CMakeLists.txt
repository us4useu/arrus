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
find_package(Us4 REQUIRED Us4Components)

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
    )
    target_link_libraries(${wrapper}
       PRIVATE
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
        "arrus/tests/beam_test.py"
        "arrus/tests/session_test.py"
        "arrus/tests/tools.py"
        "arrus/devices/__init__.py"
        "arrus/devices/us4oem.py"
        "arrus/devices/device.py"
        "arrus/devices/hv256.py"
        "arrus/devices/probe.py"
        "arius/devices/callbacks.py"
        "arrus/devices/tests/probe_test.py"
        "arrus/utils/__init__.py"
        "arrus/utils/imaging.py"
        "arrus/utils/parameters.py"
        "arrus/utils/tests/utils_test.py"
        "arrus/utils/tests/parameters_test.py"
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
