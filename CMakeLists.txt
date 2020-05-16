cmake_minimum_required(VERSION 3.17.0)

project(arrus LANGUAGES CXX)
set(PROJECT_VERSION 0.4.0rc2)
string(TIMESTAMP CURRENT_YEAR "%Y")
set(CMAKE_CXX_STANDARD 17)

################################################################################
# Modules
################################################################################
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

################################################################################
# Global options and settings
################################################################################
set(ARRUS_DOCS_INSTALL_DIR docs)
set(ARRUS_MATLAB_INSTALL_DIR matlab)
set(ARRUS_PYTHON_INSTALL_DIR python)

option(ARRUS_BUILD_PY "Build python API." OFF)
option(ARRUS_BUILD_MATLAB "Build MATLAB API." OFF)
option(ARRUS_BUILD_DOCS "Build documentation." OFF)
option(ARRUS_RUN_TESTS "Run all tests builded packages." OFF)
option(ARRUS_EMBED_DEPS "Embed dependencies (like us4r dlls) into the output package." OFF)

option(ARRUS_BUILD_SWIG "Build swig wrappers. \
NOTE: this should turned on in order to build proper API. Turn it off \
only for testing and development, for example to work on currently unsupported \
platforms." ON)
if(NOT ARRUS_BUILD_SWIG)
    message(WARNING "BUILDING PACKAGE WITHOUT LOW-LEVEL API WRAPPERS!")
endif()

################################################################################
# Sub-projects
################################################################################
add_subdirectory(core)
if(ARRUS_RUN_TESTS)
    enable_testing()
endif()
if(ARRUS_BUILD_PY)
    add_subdirectory(api/python)
endif()
if(ARRUS_BUILD_MATLAB)
    add_subdirectory(api/matlab)
endif()
if(ARRUS_BUILD_DOCS)
    add_subdirectory(docs)
endif()

################################################################################
# General docs
################################################################################
configure_file(VERSION.rst.in VERSION.rst)
install(
        FILES ${CMAKE_CURRENT_BINARY_DIR}/VERSION.rst
        DESTINATION .
)
################################################################################
# Embed external dependencies
################################################################################
if(ARRUS_EMBED_DEPS)
    message("ROOT DIR: ${Us4_ROOT_DIR}")
    install(
        DIRECTORY
            ${Us4_ROOT_DIR}/lib64/
        DESTINATION
            lib64
    )
    install(
       FILES
            # TODO(pjarosik) make it more general (will not work for OS!=win)
            ${Us4_ROOT_DIR}/matlab/Us4MEX.mexw64
       DESTINATION
            matlab/arrus
    )
    install(
        FILES
            # TODO(pjarosik) make it more general (will not work for OS!=win)
            ${Us4_ROOT_DIR}/bin/us4OEMFirmwareUpdate.exe
        DESTINATION
            bin
    )
endif()

