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
        OUTFILE_DIR arius/devices
        SOURCES wrappers/${wrapper}.i
    )
    set_property(
        TARGET ${wrapper}
        PROPERTY SWIG_USE_TARGET_INCLUDE_DIRECTORIES ON
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
# Python wheel
################################################################################
# <- pyd
# <- impl
# -> whl file



