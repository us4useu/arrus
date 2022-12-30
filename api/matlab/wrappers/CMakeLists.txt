set(TARGET_NAME arrus_mex_object_wrapper)
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
    ${ARRUS_ROOT_DIR}/arrus/common/compiler.h
    ${ARRUS_ROOT_DIR}/arrus/common/asserts.h
    ${ARRUS_ROOT_DIR}/arrus/common/format.h
    common.h
    convert.h
    MexContext.h
    MexFunction.cpp
    MexFunction.h
    MatlabClassImpl.h
    Ptr.h
    session/SessionClassImpl.h
    framework/BufferClassImpl.h
    framework/BufferElementClassImpl.h
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

target_compile_definitions(${TARGET_NAME}
    PRIVATE
    "BOOST_ALL_NO_LIB"
    )

install(
    TARGETS
    ${TARGET_NAME}
    DESTINATION
    ${ARRUS_MATLAB_INSTALL_DIR}
)

install(
	DIRECTORY
	${Us4_ROOT_DIR}/lib64/
	DESTINATION
	${ARRUS_MATLAB_INSTALL_DIR}
)

install(
	TARGETS
	arrus-core
	DESTINATION
	${ARRUS_MATLAB_INSTALL_DIR}
)
