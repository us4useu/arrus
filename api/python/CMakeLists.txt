################################################################################
# Python Swig wrappers
################################################################################
# - Targets
################################################################################
find_package(SWIG REQUIRED)
include(UseSWIG)
find_package(PythonLibs REQUIRED)
set(Us4Components
    US4OEM
    HV256
    DBARLite
)
set(Us4Wrappers
    iarius
    ihv256
    idbarlite
)
find_package(Us4 REQUIRED Us4Components)

foreach(wrapper component IN ZIP_LISTS Us4Wrappers Us4Components)
    set_property(SOURCE wrappers/${wrapper}.i PROPERTY CPLUSPLUS ON)
    swig_add_library(${wrapper}
        TYPE SHARED
        LANGUAGE PYTHON
        OUTPUT_DIR arius/devices
        OUTFILE_DIR wrappers
        SOURCES wrappers/${wrapper}.i
    )
    set_target_properties(${wrapper}
        PROPERTIES
            SWIG_USE_TARGET_INCLUDE_DIRECTORIES ON
            RUNTIME_OUTPUT_DIRECTORY
            "${CMAKE_CURRENT_BINARY_DIR}/arius/devices/$<$<CONFIG:Debug>:>"
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
################################################################################
# Python packaging
################################################################################
set(SOURCE_FILES
        "arius/beam.py"
        "arius/beam_test.py"
        "arius/interface.py"
        "arius/session.py"
        "arius/session_test.py"
        "arius/test_tools.py"
        "arius/utils_test.py"
        "arius/devices/__init__.py"
        "arius/devices/arius.py"
        "arius/devices/device.py"
        "arius/devices/hv256.py"
        "arius/devices/probe.py"
        "arius/devices/probe_test.py"
        "arius/utils/__init__.py"
        "arius/utils/imaging.py"
        "arius/utils/parameters.py"
        "arius/utils/parameters_test.py"
)

find_package(PythonInterp REQUIRED)
set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
set(SETUP_PY_OUT "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
set(TIMESTAMP "${CMAKE_CURRENT_BINARY_DIR}/timestamp")

configure_file(${SETUP_PY_IN} ${SETUP_PY_OUT})

add_custom_command(OUTPUT ${TIMESTAMP}
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/arius ${CMAKE_CURRENT_BINARY_DIR}/arius
    COMMAND ${CMAKE_COMMAND} -E touch ${TIMESTAMP}
    COMMAND ${PYTHON_EXECUTABLE} ${SETUP_PY_OUT} sdist bdist_wheel
    DEPENDS ${SOURCE_FILES} iarius ihv256 idbarlite
)

add_custom_target(python-whl ALL DEPENDS ${TIMESTAMP})



