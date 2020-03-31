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

################################################################################
# Sub-projects
################################################################################
if(BUILD_PY)
    add_subdirectory(api/python)
endif()
if(BUILD_DOCS)
    add_subdirectory(docs)
endif()

