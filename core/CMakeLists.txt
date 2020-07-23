set(TARGET_NAME arrus-core)

################################################################################
# Source files
################################################################################
set(SOURCE_FILES
    "types.h"
    "common/hash.h"
    "common/compiler.h"
    "common/exceptions.h"
    "common/asserts.h"
    "common/format.h"
    "devices/common.h"
    "devices/Device.h"
    "devices/us4oem/Us4OEMSettings.h"
    "session/SystemSettings.h"
    "session/SessionSettings.h"
    "session/Session.h"
    "session/Session.cpp"
)

# Protobuf source files
#protobuf_generate_cpp(PROTO_SRC PROTO_HDRS protobuf/settings/SessionSettings.proto)
#message("GENERATED PROTO: ${PROTO_SRC} ${PROTO_HDRS}")
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
    ${CMAKE_CURRENT_BINARY_DIR}
)

################################################################################
# Dependencies
################################################################################
target_link_libraries(${TARGET_NAME}
    PRIVATE
    Boost::Boost
    Protobuf::Protobuf
    fmt::fmt
)

################################################################################
# Tests
################################################################################
if(ARRUS_RUN_TESTS)
#    find_package(GTest REQUIRED)
#    add_executable(ttt session/tests/test.cpp ${PROTO_SRC} ${PROTO_HDRS})
#    target_link_libraries(ttt GTest::GTest Boost::Boost Protobuf::Protobuf)
#    target_include_directories(
#        ttt
#        PRIVATE
#            ${CMAKE_CURRENT_SOURCE_DIR}/..
#            ${CMAKE_CURRENT_BINARY_DIR}/..
#            ${CMAKE_CURRENT_BINARY_DIR} # Required for pb.h files.
#    )
#    add_test(NAME test1 COMMAND ttt)
endif()
