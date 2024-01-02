set(TARGET_NAME arrus-core)

################################################################################
# protobuf
################################################################################
protobuf_generate_cpp(PROTO_SRC PROTO_HDRS
    io/proto/Dictionary.proto
    io/proto/session/SessionSettings.proto
    io/proto/common/IntervalInteger.proto
    io/proto/common/IntervalDouble.proto
    io/proto/common/LinearFunction.proto
    io/proto/devices/probe/ProbeModel.proto
    io/proto/devices/us4r/ProbeAdapterModel.proto
    io/proto/devices/us4r/ProbeToAdapterConnection.proto
    io/proto/devices/us4r/RxSettings.proto
    io/proto/devices/us4r/Us4OEMSettings.proto
    io/proto/devices/us4r/HVSettings.proto
    io/proto/devices/us4r/Us4RSettings.proto
    io/proto/devices/file/FileSettings.proto
    io/proto/devices/us4r/IOAddress.proto
    io/proto/devices/us4r/IOCapability.proto
    io/proto/devices/us4r/IOSettings.proto
    io/proto/devices/us4r/Settings.proto
    io/proto/devices/us4r/Bitstream.proto
    )
################################################################################
# Target
################################################################################
set(SRC_FILES
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
    api/devices/Ultrasound.h
    api/devices/File.h
    api/session/Session.h
    api/session/SessionSettings.h
    api/ops/us4r/Pulse.h

    common/hash.h
    common/collections.h

    common/validation.h
    common/logging.h
    common/logging.cpp

    api/common/Logger.h
    api/common/LogSeverity.h
    api/common/LoggerFactory.h
    ../common/compiler.h
    ../common/asserts.h
    ../common/format.h

    devices/utils.h
    devices/DeviceId.cpp
    devices/DeviceId.h
    devices/SettingsValidator.h
    devices/TxRxParameters.h
    devices/TxRxParameters.cpp


    devices/us4r/us4oem/Us4OEMFactory.h
    devices/us4r/us4oem/Us4OEMFactoryImpl.h
    devices/us4r/us4oem/Us4OEMImpl.h
    devices/us4r/us4oem/Us4OEMImpl.cpp
    devices/us4r/us4oem/Us4OEMSettingsValidator.h
    devices/us4r/us4oem/Us4OEMSettings.h
    devices/us4r/us4oem/Us4OEMSettings.cpp

    devices/us4r/Us4RFactory.h
    devices/us4r/Us4RFactoryImpl.h
    devices/us4r/Us4RImpl.h
    devices/us4r/Us4RSettingsValidator.h
    devices/us4r/Us4RSettingsConverter.h
    devices/us4r/Us4RSettingsConverterImpl.h
    devices/us4r/Us4RSettings.h
    devices/us4r/Us4RSettings.cpp
    devices/us4r/RxSettings.h
    devices/us4r/RxSettings.cpp

    devices/us4r/probeadapter/ProbeAdapterFactory.h
    devices/us4r/probeadapter/ProbeAdapterFactoryImpl.h
    devices/us4r/probeadapter/ProbeAdapterImpl.h
    devices/us4r/probeadapter/ProbeAdapterImpl.cpp
    devices/us4r/probeadapter/ProbeAdapterSettingsValidator.h
    devices/us4r/probeadapter/ProbeAdapterSettings.h
    devices/us4r/probeadapter/ProbeAdapterSettings.cpp

    devices/probe/ProbeFactory.h
    devices/probe/ProbeFactoryImpl.h
    devices/probe/ProbeImpl.h
    devices/probe/ProbeImpl.cpp
    devices/probe/ProbeSettingsValidator.h
    devices/probe/ProbeSettings.h
    devices/probe/ProbeSettings.cpp
    devices/probe/ProbeModel.h
    devices/probe/ProbeModel.cpp

    devices/file/FileImpl.cpp
    devices/file/FileImpl.h
    devices/file/FileSettings.cpp

    api/io/settings.h
    io/settings.cpp
    io/validators/ProbeModelProtoValidator.h
    io/validators/ProbeAdapterModelProtoValidator.h
    io/validators/RxSettingsProtoValidator.h
    io/validators/SessionSettingsProtoValidator.h
    io/validators/DictionaryProtoValidator.h
    io/validators/ProbeToAdapterConnectionProtoValidator.h
    io/SettingsDictionary.h

    session/SessionImpl.cpp
    session/SessionSettings.h
    session/SessionSettings.cpp

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
    devices/us4r/Us4ROutputBuffer.h
    devices/us4r/Us4RImpl.cpp
    devices/us4r/common.h
    devices/us4r/common.cpp

    session/SessionImpl.h
    session/SessionImpl.cpp
    common/tests.h
    common/interpolate.h
    # TODO(pjarosik) cleanup below
    api/common/macros.h
    framework/graph/Graph.h
    api/devices.h
    api/framework.h
    api/ops/us4r/tgc.h
    devices/UltrasoundDevice.h
    api/devices/us4r/HVSettings.h
    api/devices/us4r/HVModelId.h
    devices/us4r/hv/HighVoltageSupplier.h
    devices/us4r/hv/HighVoltageSupplier.h
    devices/us4r/hv/HighVoltageSupplierFactory.h
    devices/us4r/hv/HighVoltageSupplier.cpp
    devices/us4r/hv/HighVoltageSupplierFactoryImpl.h
    devices/us4r/hv/HighVoltageSupplierFactoryImpl.cpp
    api/devices/us4r/FrameChannelMapping.h
    devices/us4r/FrameChannelMappingImpl.cpp
    devices/us4r/FrameChannelMappingImpl.h
    common/aperture.h
    external/eigen/Tensor.h
    devices/us4r/us4oem/Us4OEMImplBase.h
    devices/us4r/probeadapter/ProbeAdapterImplBase.h
    devices/probe/ProbeImplBase.h
    external/eigen/Dense.h
    ../common/utils.h
    devices/us4r/DataTransfer.h
    devices/us4r/us4oem/Us4OEMBuffer.h
    devices/us4r/Us4RBuffer.h
    api/ops/us4r/Scheme.h
    api/framework/Buffer.h
    api/framework/NdArray.h
    api/session/UploadResult.h
    api/framework/DataBufferSpec.h
    api/framework/Buffer.h
    api/session/Metadata.h
    api/framework/DataBuffer.h
    api/framework/FifoBuffer.h
    api/ops/us4r/DigitalDownConversion.h
    ops/us4r/DigitalDownConversion.cpp
    ops/us4r/Scheme.cpp
    cfg/default.h.in
    common/LogSeverity.cpp
    api/arrus.h
    api/session.h
    api/common.h api/io.h
    devices/us4r/validators/RxSettingsValidator.h
    devices/us4r/Us4OEMDataTransferRegistrar.h)

set_source_files_properties(${SRC_FILES} PROPERTIES COMPILE_FLAGS
    "${ARRUS_CPP_STRICT_COMPILE_OPTIONS}")

# We do not use strict compile options (-wall, /w4, etc.) for auto generated files,
# because some of those files generate compile warnings (e.g. protobuf files).

add_library(${TARGET_NAME} SHARED ${SRC_FILES} ${PROTO_SRC} ${PROTO_HDRS})
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
    ${CMAKE_CURRENT_BINARY_DIR})

################################################################################
# Dependencies
################################################################################
target_link_libraries(${TARGET_NAME}
    PRIVATE
    Us4::US4OEM
    Us4::HV256
    Us4::DBARLite
    Us4::DBARLitePcie
    Us4::Us4RDBAR
    Us4::Us4RPSC
    Boost::Boost
    protobuf::libprotobuf
    fmt::fmt
    Eigen3::Eigen3
    Microsoft.GSL::GSL
    ${ARRUS_OS_DEPS}
)
################################################################################
# Target compile options
################################################################################
# strict compile options defined for on the source level
target_compile_options(${TARGET_NAME} PRIVATE ${ARRUS_CPP_COMMON_COMPILE_OPTIONS})
target_link_options(${TARGET_NAME} PRIVATE ${ARRUS_CPP_COMMON_LINKER_OPTIONS})
target_compile_definitions(${TARGET_NAME} PRIVATE
    ${ARRUS_CPP_COMMON_COMPILE_DEFINITIONS}
    # MSVC dll export declspec
    ARRUS_CPP_API_BUILD_STAGE)
################################################################################
# Tests
################################################################################
if (ARRUS_RUN_TESTS)
    find_package(GTest REQUIRED)

    set(ARRUS_CORE_DEVICES_TESTS_SRCS devices/DeviceId.cpp common/logging.cpp)
    # core::devices test
    create_core_test(devices/DeviceIdTest.cpp devices/DeviceId.cpp)
    create_core_test(devices/utilsTest.cpp)
    create_core_test(devices/us4r/us4oem/Us4OEMSettingsValidatorTest.cpp "${ARRUS_CORE_DEVICES_TESTS_SRCS}")
    create_core_test(devices/us4r/probeadapter/ProbeAdapterSettingsValidatorTest.cpp devices/DeviceId.cpp)
    create_core_test(devices/probe/ProbeSettingsValidatorTest.cpp devices/DeviceId.cpp)
    set(US4OEM_FACTORY_IMPL_TEST_DEPS devices/DeviceId.cpp common/logging.cpp devices/us4r/us4oem/Us4OEMImpl.cpp
        devices/TxRxParameters.cpp devices/us4r/FrameChannelMappingImpl.cpp
        ops/us4r/DigitalDownConversion.cpp)
    create_core_test(devices/us4r/us4oem/Us4OEMFactoryImplTest.cpp "${US4OEM_FACTORY_IMPL_TEST_DEPS}")
    create_core_test(devices/us4r/Us4RSettingsConverterImplTest.cpp devices/DeviceId.cpp)
    create_core_test(devices/us4r/external/ius4oem/IUs4OEMInitializerImplTest.cpp)
    create_core_test(devices/us4r/commonTest.cpp "devices/us4r/common.cpp;devices/TxRxParameters.cpp")

    set(US4OEM_IMPL_TEST_DEPS common/logging.cpp devices/us4r/us4oem/Us4OEMImpl.cpp
        devices/us4r/common.cpp devices/TxRxParameters.cpp devices/DeviceId.cpp
        devices/us4r/FrameChannelMappingImpl.cpp
        ops/us4r/DigitalDownConversion.cpp)
    create_core_test(devices/us4r/us4oem/Us4OEMImplTest.cpp "${US4OEM_IMPL_TEST_DEPS}")

    set(ADAPTER_IMPL_TEST_DEPS common/logging.cpp devices/us4r/probeadapter/ProbeAdapterImpl.cpp
        devices/us4r/common.cpp
        devices/TxRxParameters.cpp devices/DeviceId.cpp devices/us4r/FrameChannelMappingImpl.cpp
        ops/us4r/DigitalDownConversion.cpp)
    create_core_test(devices/us4r/probeadapter/ProbeAdapterImplTest.cpp "${ADAPTER_IMPL_TEST_DEPS}")
    create_core_test(devices/us4r/Us4OEMDataTransferRegistrarTest.cpp common/logging.cpp)
    create_core_test(devices/probe/ProbeImplTest.cpp
        "devices/probe/ProbeImpl.cpp;devices/us4r/FrameChannelMappingImpl.cpp;common/logging.cpp;devices/DeviceId.cpp")
    # core::io tests
    set(ARRUS_CORE_IO_TEST_DATA ${CMAKE_CURRENT_SOURCE_DIR}/io/test-data)
    create_core_test(io/settingsTest.cpp
        "" # no additional source files
        "protobuf::libprotobuf;arrus-core"
        "-DARRUS_TEST_DATA_PATH=\"${ARRUS_CORE_IO_TEST_DATA}\"")
endif ()

################################################################################
# Configuration
################################################################################
# Create default configuration header.
file(READ "${PROJECT_SOURCE_DIR}/arrus/cfg/default.dict" ARRUS_DEFAULT_DICT)

# Split default dictionary string into multiple parts - necessary due to
# relatively small string length limitations in MSVC.
string(LENGTH ${ARRUS_DEFAULT_DICT} ARRUS_DEFAULT_DICT_LEN)
set(ARRUS_DEFAULT_DICT_CURRENT_POS 0)
set(ARRUS_DEFAULT_DICT_PART_SIZE 16000)
set(ARRUS_DEFAULT_DICT_VALUE "")
while(${ARRUS_DEFAULT_DICT_CURRENT_POS} LESS_EQUAL ${ARRUS_DEFAULT_DICT_LEN})
    string(SUBSTRING ${ARRUS_DEFAULT_DICT} ${ARRUS_DEFAULT_DICT_CURRENT_POS} ${ARRUS_DEFAULT_DICT_PART_SIZE} ARRUS_DEFAULT_DICT_PART)
    set(ARRUS_DEFAULT_DICT_VALUE "${ARRUS_DEFAULT_DICT_VALUE} R\"\"\"\"(${ARRUS_DEFAULT_DICT_PART})\"\"\"\"")
    math(EXPR ARRUS_DEFAULT_DICT_CURRENT_POS "${ARRUS_DEFAULT_DICT_CURRENT_POS} + ${ARRUS_DEFAULT_DICT_PART_SIZE}" OUTPUT_FORMAT DECIMAL)
endwhile()
# Splitting done
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cfg/default.h.in ${CMAKE_CURRENT_BINARY_DIR}/cfg/default.h)


################################################################################
# Installation
################################################################################
install(
    TARGETS
    ${TARGET_NAME}
    DESTINATION
    ${ARRUS_LIB_INSTALL_DIR}
)

install(
    DIRECTORY
    api
    DESTINATION
    ${ARRUS_INCLUDE_INSTALL_DIR}/arrus/core
)

install(
    DIRECTORY
    examples
    DESTINATION
    .
)
################################################################################
# Examples
################################################################################
add_executable(pwi-example examples/PwiExample.cpp)
target_link_libraries(pwi-example PRIVATE arrus-core)
target_include_directories(pwi-example PRIVATE ${ARRUS_ROOT_DIR})
#
#add_executable(demod-example examples/AfeDDCExample_v2.cpp)
#target_link_libraries(demod-example PRIVATE arrus-core Boost::Boost)
#target_include_directories(demod-example PRIVATE ${ARRUS_ROOT_DIR})
#configure_file(examples/fir.txt fir.txt COPYONLY)
#configure_file(examples/AfeDDCExample_v2_FirGen.py AfeDDCExample_v2_FirGen.py COPYONLY)
