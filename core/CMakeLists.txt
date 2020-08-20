set(TARGET_NAME arrus-core)

# Protobuf source files
#protobuf_generate_cpp(PROTO_SRC PROTO_HDRS protobuf/settings/SessionSettings.proto)
#message("GENERATED PROTO: ${PROTO_SRC} ${PROTO_HDRS}")
################################################################################
# Target
################################################################################
add_library(${TARGET_NAME} SHARED

    api/common/exceptions.h
    api/common/logging.h
    api/common/types.h
    api/common/Tuple.h
    api/common/Interval.h
    api/devices/DeviceWithComponents.h
    api/devices/Device.h
    api/devices/TriggerGenerator.h
    api/devices/DeviceId.h
    api/devices/probe/Probe.h
    api/devices/probe/ProbeModel.h
    api/devices/probe/ProbeModelId.h
    api/devices/probe/ProbeSettings.h
    api/devices/us4r/ProbeAdapter.h
    api/devices/us4r/ProbeAdapterModelId.h
    api/devices/us4r/ProbeAdapterSettings.h
    api/devices/us4r/Us4OEM.h
    api/devices/us4r/Us4OEMSettings.h
    api/devices/us4r/Us4R.h
    api/devices/us4r/Us4RSettings.h
    api/devices/us4r/RxSettings.h
    api/session/Session.h
    api/session/SessionSettings.h

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

    devices/utils.h
    devices/DeviceId.cpp
    devices/DeviceIdHasher.h
    devices/DeviceSettingsValidator.h

    devices/us4r/us4oem/Us4OEMFactory.h
    devices/us4r/us4oem/Us4OEMFactoryImpl.h
    devices/us4r/us4oem/Us4OEMImpl.h
    devices/us4r/us4oem/Us4OEMSettingsValidator.h

    devices/us4r/Us4RFactory.h
    devices/us4r/Us4RFactoryImpl.h
    devices/us4r/Us4RImpl.h
    devices/us4r/Us4RSettingsValidator.h
    devices/us4r/Us4RSettingsConverter.h
    devices/us4r/Us4RSettingsConverterImpl.h

    devices/us4r/probeadapter/ProbeAdapterFactory.h
    devices/us4r/probeadapter/ProbeAdapterFactoryImpl.h
    devices/us4r/probeadapter/ProbeAdapterImpl.h
    devices/us4r/probeadapter/ProbeAdapterSettingsValidator.h

    devices/probe/ProbeFactory.h
    devices/probe/ProbeFactoryImpl.h
    devices/probe/ProbeImpl.h
    devices/probe/ProbeSettingsValidator.h

    session/SessionImpl.cpp

    devices/us4r/external/ius4oem/IUs4OEMFactory.h
    devices/us4r/external/ius4oem/IUs4OEMFactoryImpl.h
    devices/us4r/external/ius4oem/LNAGainValueMap.h
    devices/us4r/external/ius4oem/PGAGainValueMap.h
    devices/us4r/external/ius4oem/ActiveTerminationValueMap.h
    devices/us4r/external/ius4oem/LPFCutoffValueMap.h
    devices/us4r/external/ius4oem/DTGCAttenuationValueMap.h
    devices/us4r/external/ius4oem/Us4RLoggerWrapper.h
    devices/us4r/external/ius4oem/IUs4OEMInitializer.h
    devices/us4r/external/ius4oem/IUs4OEMInitializerImpl.h

    session/SessionImpl.h
    session/SessionImpl.cpp
)

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
    range-v3::range-v3
)

################################################################################
# Tests
################################################################################
if(ARRUS_RUN_TESTS)
    find_package(GTest REQUIRED)
    create_core_test(devices/DeviceIdTest.cpp)
    create_core_test(devices/utilsTest.cpp)
    create_core_test(devices/us4r/us4oem/Us4OEMSettingsValidatorTest.cpp)
    create_core_test(devices/us4r/probeadapter/ProbeAdapterSettingsValidatorTest.cpp)
    create_core_test(devices/probe/ProbeSettingsValidatorTest.cpp)
    create_core_test(devices/us4r/us4oem/Us4OEMFactoryImplTest.cpp)
    create_core_test(devices/us4r/Us4RFactoryImplTest.cpp)
    create_core_test(devices/us4r/Us4RSettingsConverterImplTest.cpp)
    create_core_test(devices/us4r/external/ius4oem/IUs4OEMInitializerImplTest.cpp)
endif()
