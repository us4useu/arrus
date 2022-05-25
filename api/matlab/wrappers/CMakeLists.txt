set(TARGET_NAME arrus_mex_object_wrapper)

message("Building mex with name ${TARGET_NAME}")

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
    MODULE
    SRC
        ${ARRUS_ROOT_DIR}/arrus/common/compiler.h
        ${ARRUS_ROOT_DIR}/arrus/common/asserts.h
        ${ARRUS_ROOT_DIR}/arrus/common/format.h
        ${ARRUS_ROOT_DIR}/arrus/common/logging/impl/LoggerImpl.h
        ${ARRUS_ROOT_DIR}/arrus/common/logging/impl/Logging.cpp
        ${ARRUS_ROOT_DIR}/arrus/common/logging/impl/Logging.h
        ${ARRUS_ROOT_DIR}/arrus/common/logging/impl/LogSeverity.cpp
        common.h
        convert.h
        DefaultMexObjectManager.h
        MexContext.h
        MexFunction.cpp
        MexFunction.h
        MexObjectManager.h
        MexObjectWrapper.h
#        session/SessionWrapper.h
#        session/convertSessionSettings.h
#        devices/convertDeviceId.h
#        devices/us4r/convertUs4RSettings.h
#        devices/us4r/convertUs4OEMSettings.h
#        devices/us4r/convertProbeAdapterSettings.h
#        devices/us4r/convertProbeAdapterModelId.h
#        devices/probe/convertProbeModel.h
#        devices/probe/convertProbeModelId.h
#        devices/probe/convertProbeSettings.h
#        devices/us4r/convertRxSettings.h
    MatlabStdoutBuffer.h
    LINK_TO
        arrus-core
        Boost::Boost
        fmt::fmt
        Microsoft.GSL::GSL
)
################################################################################
# Include directories
################################################################################
target_include_directories(
    ${TARGET_NAME}
    PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${ARRUS_ROOT_DIR}
)

################################################################################
# Compile and link options
################################################################################
target_compile_definitions(${TARGET_NAME}
    PRIVATE
    # Mute std::unique() deprecation warning from matlab api headers.
    "_SILENCE_CXX17_SHARED_PTR_UNIQUE_DEPRECATION_WARNING"
)

install(
        TARGETS
        ${TARGET_NAME}
        DESTINATION
        ${ARRUS_MATLAB_INSTALL_DIR}/+arrus
)

