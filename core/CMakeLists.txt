set(TARGET_NAME arrus-core)

################################################################################
# Source files
################################################################################
set(SOURCE_FILES
    # API
    api/common/exceptions.h
    api/common/logging.h
    api/common/types.h
    api/common/tgc.h
    api/devices/Device.h
    api/devices/UltrasoundDevice.h
    api/devices/TriggerGenerator.h
    api/devices/DeviceId.h
    api/devices/probe/Probe.h
    api/devices/probe/ProbeSettings.h
    api/devices/probeadapter/ProbeAdapter.h
    api/devices/probeadapter/ProbeAdapterSettings.h
    api/devices/us4oem/Us4OEM.h
    api/devices/us4oem/Us4OEMSettings.h
    api/devices/us4r/Us4R.h
    api/devices/us4r/Us4RSettings.h

    common/hash.h
    common/compiler.h
    common/collections.h
    common/asserts.h
    common/format.h
    common/validation.h
    common/logging.h
    common/logging.cpp

    ../common/logging/Logger.h
    ../common/logging/LogSeverity.h
    ../common/logging/LoggerFactory.h

    devices/DeviceId.cpp
    devices/DeviceIdHasher.h
    devices/DeviceSettingsValidator.h

    devices/us4oem/Us4OEMFactory.h
    devices/us4oem/Us4OEMFactoryImpl.h
    devices/us4oem/Us4OEMImpl.h
    devices/us4oem/Us4OEMImpl.cpp
    devices/us4oem/Us4OEMSettingsValidator.h

    devices/us4r/Us4RFactory.h
    devices/us4r/Us4RFactoryImpl.h
    devices/us4r/Us4RImpl.h
    devices/us4r/Us4RImpl.cpp
    devices/us4r/Us4RSettingsValidator.h

    devices/probeadapter/ProbeAdapterFactory.h

    devices/probe/ProbeFactory.h

    session/Session.h
    session/Session.cpp

    external/ius4oem/IUs4OEMFactory.h
    external/ius4oem/IUs4OEMFactoryImpl.h
    external/ius4oem/LNAGainValueMap.h
    external/ius4oem/PGAGainValueMap.h
    external/ius4oem/ActiveTerminationValueMap.h
    external/ius4oem/LPFCutoffValueMap.h
    external/ius4oem/DTGCAttenuationValueMap.h
    external/ius4oem/Us4RLoggerWrapper.h
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
    ${ARRUS_ROOT_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
)

################################################################################
# Dependencies
################################################################################
target_link_libraries(${TARGET_NAME}
    PRIVATE
    Us4::US4OEM
    Us4::HV256
    Us4::DBARLite
    Boost::Boost
    Protobuf::Protobuf
    fmt::fmt
)

################################################################################
# Tests
################################################################################
if(ARRUS_RUN_TESTS)
    find_package(GTest REQUIRED)
    create_core_test(devices/DeviceIdTest.cpp)
    create_core_test(devices/us4oem/Us4OEMSettingsValidatorTest.cpp)
    create_core_test(devices/us4oem/Us4OEMFactoryImplTest.cpp)
endif()
