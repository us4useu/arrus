function(create_core_test test_src)
    # Optional arguments.
    if(${ARGC} GREATER 1)
        set(other_srcs ${ARGV1})
    endif()
    if(${ARGC} GREATER 2)
        set(other_deps ${ARGV2})
    endif()
    if(${ARGC} GREATER 2)
        set(compile_definitions ${ARGV3})
    endif()

    # TODO(pjarosik) make the below a parameter
    if(NOT DEFINED ARRUS_CPP_COMMON_COMPILE_OPTIONS)
        message(FATAL_ERROR "ARRUS_CPP_COMMON_COMPILE_OPTIONS must be set for test targets.")
    endif()

    # replace / in test_src with _
    get_filename_component(target_name_file ${test_src} NAME_WE)
    get_filename_component(target_name_dir ${test_src} DIRECTORY)
    string(REPLACE "/" "_" target_name "${target_name_dir}/${target_name_file}")

    add_executable(${target_name}
        ${test_src}
        ../common/logging/impl/Logging.cpp
        ../common/logging/impl/LogSeverity.cpp
        ${other_srcs}
    )
    target_link_libraries(${target_name}
        GTest::GTest
        Boost::Boost
        fmt::fmt
        range-v3::range-v3
        ${other_deps})
    target_include_directories(
        ${target_name}
        PRIVATE
        ${ARRUS_ROOT_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}/..
        ${CMAKE_CURRENT_BINARY_DIR} # Required for pb.h files (protobuf).
        ${Us4_INCLUDE_DIR} # Required to mock us4 devices. TODO(pjarosik) do not depend tests on external libraries
    )
    target_compile_options(${target_name} PRIVATE ${ARRUS_CPP_COMMON_COMPILE_OPTIONS})
    target_compile_definitions(${target_name} PRIVATE
        ARRUS_CORE_UNIT_TESTS
        _SILENCE_CXX17_ALLOCATOR_VOID_DEPRECATION_WARNING
        ${compile_definitions})
    add_test(NAME ${test_src} COMMAND ${target_name})

    prepend_env_path(ARRUS_TESTS_ENV_PATH ${Us4_LIB_DIR})
    set_tests_properties(${test_src} PROPERTIES ENVIRONMENT "PATH=${ARRUS_TESTS_ENV_PATH}")
endfunction()