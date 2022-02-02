# Option: Us4_ROOT_DIR: a directory, where lib64 and include files are located.

if(NOT DEFINED Us4_ROOT_DIR)
    message(FATAL_ERROR "Us4_ROOT_DIR should be provided.")
endif()

find_path(Us4_INCLUDE_DIR
        NAMES ius4oem.h
        PATHS "${Us4_ROOT_DIR}/include"
        PATH_SUFFIXES Us4
        NO_CMAKE_ENVIRONMENT_PATH
        NO_SYSTEM_ENVIRONMENT_PATH
)
set(Us4_LIBRARY_DIR "${Us4_ROOT_DIR}/lib64")
find_library(Us4_US4OEM_LIBRARY
        NAMES Us4OEM
        PATHS ${Us4_LIBRARY_DIR}
        NO_CMAKE_ENVIRONMENT_PATH
        NO_SYSTEM_ENVIRONMENT_PATH
)
find_library(Us4_HV256_LIBRARY
        NAMES HV256
        PATHS ${Us4_LIBRARY_DIR}
        NO_CMAKE_ENVIRONMENT_PATH
        NO_SYSTEM_ENVIRONMENT_PATH
)
find_library(Us4_DBARLite_LIBRARY
        NAMES DBARLite
        PATHS ${Us4_LIBRARY_DIR}
        NO_CMAKE_ENVIRONMENT_PATH
        NO_SYSTEM_ENVIRONMENT_PATH
)

# Read required version from include/versions.h file
file(READ "${Us4_ROOT_DIR}/include/versions.h" Us4OEM_VERSIONS)

# API version
string(REGEX MATCH "US4R_API_VERSION[ ]+([0-9]\\.[0-9]\\.[0-9])"
       US4API_VERSION_MATCH
       ${Us4OEM_VERSIONS}
)
if("${US4API_VERSION_MATCH}" STREQUAL "")
    message(FATAL_ERROR "Couldn't read US4R_API_VERSION from us4r distribution.")
endif()
set(Us4_VERSION ${CMAKE_MATCH_1})
message("Found Us4R version ${Us4_VERSION}")

# Firmware version
string(REGEX MATCH "US4OEM_FIRMWARE_VERSION \\{(.*)\\}"
        US4OEM_FIRMWARE_VERSION_MATCH
        ${Us4OEM_VERSIONS}
)

if("${US4OEM_FIRMWARE_VERSION_MATCH}" STREQUAL "")
    message(FATAL_ERROR "Couldn't read US4OEM_FIRMWARE_VERSION from us4r distribution.")
endif()
set(Us4OEM_FIRMWARE_VERSION ${CMAKE_MATCH_1})
message("Required module firmware version: ${Us4OEM_FIRMWARE_VERSION}")

# TX Firmware version
string(REGEX MATCH "US4OEM_TX_FIRMWARE_VERSION \\{(.*)\\}"
        US4OEM_TX_FIRMWARE_VERSION_MATCH
        ${Us4OEM_VERSIONS}
)
if("${US4OEM_TX_FIRMWARE_VERSION_MATCH}" STREQUAL "")
    message(FATAL_ERROR "Couldn't read US4OEM_TX_FIRMWARE_VERSION from us4r distribution.")
endif()
set(Us4OEM_TX_FIRMWARE_VERSION ${CMAKE_MATCH_1})
message("Required module tx firmware version: ${Us4OEM_TX_FIRMWARE_VERSION}")


include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
        Us4
        FOUND_VAR Us4_FOUND
        REQUIRED_VARS
        Us4_US4OEM_LIBRARY
        Us4_HV256_LIBRARY
        Us4_DBARLite_LIBRARY
        Us4_INCLUDE_DIR
        VERSION_VAR Us4_VERSION
)

if(Us4_FOUND AND NOT TARGET Us4::US4OEM)
    add_library(Us4::US4OEM UNKNOWN IMPORTED)
    set_target_properties(Us4::US4OEM PROPERTIES
            IMPORTED_LOCATION "${Us4_US4OEM_LIBRARY}"
            INTERFACE_COMPILE_OPTIONS "${PC_Us4_CFLAGS_OTHER}"
            INTERFACE_INCLUDE_DIRECTORIES "${Us4_INCLUDE_DIR}"
    )
endif()

if(Us4_FOUND AND NOT TARGET Us4::HV256)
    add_library(Us4::HV256 UNKNOWN IMPORTED)
    set_target_properties(Us4::HV256 PROPERTIES
            IMPORTED_LOCATION "${Us4_HV256_LIBRARY}"
            INTERFACE_COMPILE_OPTIONS "${PC_Us4_CFLAGS_OTHER}"
            INTERFACE_INCLUDE_DIRECTORIES "${Us4_INCLUDE_DIR}"
    )
endif()

if(Us4_FOUND AND NOT TARGET Us4::DBARLite)
    add_library(Us4::DBARLite UNKNOWN IMPORTED)
    set_target_properties(Us4::DBARLite PROPERTIES
            IMPORTED_LOCATION "${Us4_DBARLite_LIBRARY}"
            INTERFACE_COMPILE_OPTIONS "${PC_Us4_CFLAGS_OTHER}"
            INTERFACE_INCLUDE_DIRECTORIES "${Us4_INCLUDE_DIR}"
    )
endif()
