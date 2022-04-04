cmake_minimum_required(VERSION 3.17)
project(arrus-cpp-examples)

set(CMAKE_CXX_STANDARD 17)

# TODO set path to us4r-lite headers and libraries
set(ARRUS_INCLUDE_PATH "/path/to/set/include")
set(ARRUS_LIB_PATH "/path/to/set/lib64")

add_executable(sta-example StaExample.cpp)
target_link_directories(sta-example PRIVATE ${ARRUS_LIB_PATH})
target_link_libraries(sta-example PRIVATE arrus-core)
target_include_directories(sta-example PRIVATE ${ARRUS_INCLUDE_PATH})

add_executable(pwi-example PwiExample.cpp)
target_link_directories(pwi-example PRIVATE ${ARRUS_LIB_PATH})
target_link_libraries(pwi-example PRIVATE arrus-core)
target_include_directories(pwi-example PRIVATE ${ARRUS_INCLUDE_PATH})

add_executable(demod-example AfeDDCExample.cpp)
target_link_directories(demod-example PRIVATE ${ARRUS_LIB_PATH})
target_link_libraries(demod-example PRIVATE arrus-core)
target_include_directories(demod-example PRIVATE ${ARRUS_INCLUDE_PATH})
