function(create_core_test test_src)
    # Optional arguments.
    if(${ARGC} GREATER 1)
        set(other_srcs ${ARGV1})
    endif()
    if(${ARGC} GREATER 2)
        set(other_deps ${ARGV2})
    endif()

    # replace / in test_src with _
    get_filename_component(target_name_file ${test_src} NAME_WE)
    get_filename_component(target_name_dir ${test_src} DIRECTORY)
    string(REPLACE "/" "_" target_name "${target_name_dir}/${target_name_file}")

    add_executable(${target_name} ${test_src} ${other_srcs})
    target_link_libraries(${target_name} arrus-core GTest::GTest Boost::Boost fmt::fmt ${other_deps})
    target_include_directories(
        ${target_name}
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_BINARY_DIR}/..
        ${CMAKE_CURRENT_BINARY_DIR} # Required for pb.h files (protobuf).
        ${Us4_INCLUDE_DIR} # Required to mock us4 devices. TODO(pjarosik) do not depend tests on external libraries
    )
    add_test(NAME ${test_src} COMMAND ${target_name})
endfunction()