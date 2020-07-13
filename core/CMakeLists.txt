set(TARGET_NAME arrus-core)

set(SOURCE_FILES
        "types.h"
        "devices/common.h"
        "devices/Us4OEMSettings.h"
        "session/SessionSettings.h"
        "session/Session.h"
        "session/Session.cpp"
)

add_library(${TARGET_NAME} SHARED ${SOURCE_FILES})

target_include_directories(${TARGET_NAME}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..
)


