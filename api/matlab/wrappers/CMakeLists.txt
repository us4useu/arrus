set(TARGET_NAME mex_object_wrapper)

################################################################################
# Source files
################################################################################
set(SOURCE_FILES
    "common.h"
    "DefaultMexObjectManager.h"
    "MexContext.h"
    "MexFunction.cpp"
    "MexFunction.h"
    "MexObjectManager.h"
    "MexObjectWrapper.h"
    "arrus/SessionWrapper.h"
)

################################################################################
# Target and dependencies
################################################################################
find_package(Matlab REQUIRED) # TODO require exact 9.5 version

if (NOT Matlab_FOUND)
    message(WARNING "Matlab not found, Us4MEX target not available.")
    return()
endif ()

matlab_add_mex(
    NAME
        ${TARGET_NAME}
    SHARED
    SRC
        ${SOURCE_FILES}
    LINK_TO
        Boost::Boost
        fmt::fmt
)
################################################################################
# Include directories
################################################################################
target_include_directories(
    ${TARGET_NAME}
    PRIVATE
        ${PROJECT_SOURCE_DIR}
)

################################################################################
# Compile and link options
################################################################################
target_compile_definitions(${TARGET_NAME}
    PRIVATE
    # Mute std::unique() deprecation warning from matlab api headers.
    "_SILENCE_CXX17_SHARED_PTR_UNIQUE_DEPRECATION_WARNING"
)

