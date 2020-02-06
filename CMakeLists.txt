cmake_minimum_required(VERSION 3.13.0 FATAL_ERROR)

project(arius-sdk LANGUAGES CXX VERSION 0.2.0)
set(CMAKE_CXX_STANDARD 17)

################################################################################
# Modules
################################################################################
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

################################################################################
# Global options and settings
################################################################################
set(ARIUS_DOCS_INSTALL_DIR docs)

################################################################################
# Sub-projects
################################################################################
add_subdirectory(python)

