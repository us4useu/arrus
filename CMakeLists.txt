cmake_minimum_required(VERSION 3.17.0)

project(arius-sdk LANGUAGES CXX VERSION 0.3.0)
string(TIMESTAMP CURRENT_YEAR "%Y")
set(CMAKE_CXX_STANDARD 17)

################################################################################
# Modules
################################################################################
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

################################################################################
# Global options and settings
################################################################################
set(ARIUS_DOCS_INSTALL_DIR docs)

option(ARIUS_BUILD_PY "Build python API." OFF)
option(ARIUS_BUILD_MATLAB "Build MATLAB API." OFF)
option(ARIUS_BUILD_DOCS "Build documentation." OFF)
option(ARIUS_RUN_TESTS "Run all tests builded packages." OFF)

option(ARIUS_BUILD_SWIG "Build swig wrappers. \
NOTE: this should turned on in order to build proper API. Turn it off \
only for testing and development, for example to work on currently unsupported \
platforms." ON)
if(NOT ARIUS_BUILD_SWIG)
    message(WARNING "BUILDING PACKAGE WITHOUT LOW-LEVEL API WRAPPERS!")
endif()

################################################################################
# Sub-projects
################################################################################
if(ARIUS_RUN_TESTS)
    enable_testing()
endif()
if(ARIUS_BUILD_PY)
    add_subdirectory(api/python)
endif()
if(ARIUS_BUILD_DOCS)
    add_subdirectory(docs)
endif()

