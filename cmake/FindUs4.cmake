# Option: Us4_ROOT_DIR: a directory, where lib64 and include files are located.
find_path(Us4_INCLUDE_DIR
        NAMES ius4oem.h
        PATHS "${Us4_ROOT_DIR}/include"
        PATH_SUFFIXES Us4
)
set(Us4_LIBRARY_DIR "${Us4_ROOT_DIR}/lib64")
find_library(Us4_US4OEM_LIBRARY
        NAMES Us4OEM
        PATHS ${Us4_LIBRARY_DIR}
)
find_library(Us4_HV256_LIBRARY
        NAMES HV256
        PATHS ${Us4_LIBRARY_DIR}
)
find_library(Us4_DBARLite_LIBRARY
        NAMES DBARLite
        PATHS ${Us4_LIBRARY_DIR}
)
set(Us4_VERSION ${PC_Us4_VERSION})

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
