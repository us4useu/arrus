set(TARGET_NAME arrus-core)

################################################################################
# Source files
################################################################################
set(SOURCE_FILES
    "types.h"
    "utils/common.h"
    "devices/common.h"
    "devices/Device.h"
    "devices/us4oem/Us4OEMSettings.h"
    "session/SystemSettings.h"
    "session/SessionSettings.h"
    "session/Session.h"
    "session/Session.cpp"
)

################################################################################
# Target
################################################################################
add_library(${TARGET_NAME} SHARED ${SOURCE_FILES})

################################################################################
# Compile definitions
################################################################################
target_compile_definitions(${TARGET_NAME}
    PRIVATE
    "BOOST_ALL_NO_LIB"
)

################################################################################
# Include directories
################################################################################
target_include_directories(${TARGET_NAME}
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

################################################################################
# Dependencies
################################################################################
target_link_libraries(${TARGET_NAME}
    PRIVATE
    Boost::Boost
)


