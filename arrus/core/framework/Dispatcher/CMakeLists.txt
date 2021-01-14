cmake_minimum_required (VERSION 3.5)

set (CMAKE_CXX_STANDARD 11)
#add_definitions("-fpermissive")

file (GLOB_RECURSE DISPATCHER_SOURCES "HeaderFiles/*.h" "SourceFiles/*.cpp" "SourceFiles/*.cu" "HeaderFiles/*.cuh")

list (REMOVE_ITEM DISPATCHER_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/SourceFiles/Model/GraphNodesLibrary/GraphNodes/PluginGraphNode/PluginGraphNode.cpp )
list (REMOVE_ITEM DISPATCHER_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/SourceFiles/Model/GraphNodesLibrary/GraphNodes/CudaKernelGraphNode.cpp )

source_group ("" FILES ${DISPATCHER_SOURCES})

add_definitions (-DBOOST_ALL_DYN_LINK)

include_directories (   "HeaderFiles"
                        "../HAL")

#include_directories (   "/usr/include"
#                        "/usr/include/boost"
#                        "/usr/include/python2.7"
#                        "${PROJECT_SOURCE_DIR}/ARRIUS"
#                        "/usr/local/cuda-8.0/targets/aarch64-linux/include")

find_package (Boost 1.58 COMPONENTS python log REQUIRED)
find_package(CUDA QUIET REQUIRED)

#add_library(NVTX SHARED IMPORTED)
#set_target_properties(NVTX PROPERTIES
#  IMPORTED_LOCATION "/usr/local/cuda-8.0/targets/aarch64-linux/lib/libnvToolsExt.so.1"
#)

set (CUDA_SEPARABLE_COMPILATION ON)

# Pass options to NVCC
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O2
    -gencode arch=compute_53,code=sm_53
    -std=c++11
    -rdc=true
    #-x cu
)

#target_link_libraries (	DISPATCHER
#                        nvrtc
#                        )
#			ARRIUS
#			${Boost_LIBRARIES}
#			${PYTHON_LIBRARIES}
#			NVTX)

cuda_add_library (DISPATCHER SHARED ${DISPATCHER_SOURCES})

CUDA_ADD_CUFFT_TO_TARGET( DISPATCHER )

install (TARGETS DISPATCHER DESTINATION "${PROJECT_SOURCE_DIR}/bin")
