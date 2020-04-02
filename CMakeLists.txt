cmake_minimum_required(VERSION 3.17.0)

project(arius-sdk LANGUAGES CXX VERSION 0.2.0)
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

option(BUILD_PY "Build python API." OFF)
option(BUILD_DOCS "Build documentation." OFF)
option(RUN_TESTS "Run all tests builded packages." OFF)

option(BUILD_SWIG "Build swig wrappers. \
NOTE: this should turned on in order to build proper API. Turn it off \
only for testing and development, for example to work on currently unsupported \
platforms." ON)
if(NOT BUILD_SWIG)
    message(WARNING "BUILDING PACKAGE WITHOUT LOW-LEVEL API WRAPPERS!")
endif()

################################################################################
# Sub-projects
################################################################################
if(BUILD_PY)
    add_subdirectory(api/python)
endif()
if(BUILD_DOCS)
    add_subdirectory(docs)
endif()

