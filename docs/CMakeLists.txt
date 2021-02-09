################################################################################
# Sphinx
################################################################################
set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR})

set(DOC_FILES
    "index.rst"
    "content/introduction.rst"
    "content/installation/index.rst"
    "content/installation/api.python.rst"
    "content/installation/api.matlab.rst"
    "content/installation/img/dev_manager.png"
    "content/installation/img/paths.png"
    "content/installation/img/result.png"
    "content/installation/img/install_result.png"
    "content/installation/img/thunderbolt.png"
    "content/installation/img/uninstall_arius_drv.png"
    "content/examples/bmode_imaging.matlab.rst"
    content/examples/mock_session_example.python.rst
    content/user_guide.python.rst
    content/definitions.python.rst
    "content/examples/img/coordinate_system.jpeg"
    "content/examples/img/apertures.jpeg"
    "content/examples/img/delays.jpeg"
    "content/img/coordinate_system.jpeg"
    "content/img/apertures.jpeg"
    "content/img/delays.jpeg"
    "content/img/scheme.jpeg"
    "content/misc/release_notes.rst"
    "content/misc/acknowledgments.rst"
    "content/misc/img/ncbr.png"
    "content/misc/img/us4us.png"
    "content/api.python.rst"
    "content/api.matlab.rst"
    "content/api.cpp.rst"
    "content/low_level_api.matlab.rst"
)
list(TRANSFORM DOC_FILES PREPEND "${SPHINX_SOURCE}/")

################################################################################
# Sphinx Docs generator.
################################################################################
function(CREATE_DOC_TARGET LANGUAGE FORMAT VENV_TARGET SPHINX_TARGET DEPS CUSTOM_OPTIONS)
    get_target_property(VENV_PY_EXECUTABLE ${VENV_TARGET} VENV_EXECUTABLE)
    get_target_property(SPHINX_EXECUTABLE ${SPHINX_TARGET} SPHINX_EXECUTABLE)

    if(FORMAT STREQUAL "pdf")
        set(FORMAT_OUT_DIR "latex")
        set(FORMAT_SPHINX_METHOD "latexpdf")
        set(FORMAT_TRACKED_FILE "arrus${LANGUAGE}.pdf")
    elseif(FORMAT STREQUAL "html")
        set(FORMAT_OUT_DIR "html")
        set(FORMAT_SPHINX_METHOD "html")
        set(FORMAT_TRACKED_FILE "index.html")
    else()
        message(FATAL_ERROR "Unsupported format: ${FORMAT}")
    endif()

    set(SPHINX_CONF_FILE_IN "${SPHINX_SOURCE}/conf.${LANGUAGE}.py.in")
    set(SPHINX_CURRENT_LANG_DIR ${CMAKE_CURRENT_BINARY_DIR}/${LANGUAGE})
    set(SPHINX_BUILD ${SPHINX_CURRENT_LANG_DIR}/sphinx)
    file(MAKE_DIRECTORY ${SPHINX_BUILD})
    set(SPHINX_LATEX_CONF_DIR ${SPHINX_CURRENT_LANG_DIR}/cfg_${FORMAT_OUT_DIR})
    set(SPHINX_LATEX_CONF_FILE_OUT ${SPHINX_LATEX_CONF_DIR}/conf.py)
    configure_file(${SPHINX_CONF_FILE_IN} ${SPHINX_LATEX_CONF_FILE_OUT})

    set(SPHINX_TRACKED_DIR "${SPHINX_BUILD}/${FORMAT_OUT_DIR}")
    set(SPHINX_TRACKED_FILE "${SPHINX_TRACKED_DIR}/${FORMAT_TRACKED_FILE}")

    set(SPHINX_CUSTOM_OPTIONS "")

    # TODO Move this outside this
    if(${LANGUAGE} STREQUAL "cpp")

    endif()

    add_custom_command(OUTPUT ${SPHINX_TRACKED_FILE}
        COMMAND
            ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_CURRENT_SOURCE_DIR}/content
                ${SPHINX_CURRENT_LANG_DIR}/content
        COMMAND
            ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_SOURCE_DIR}/index.rst
                ${SPHINX_CURRENT_LANG_DIR}/index.rst
        COMMAND
            ${VENV_PY_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/restructure_tree.py
                --root_dir ${SPHINX_CURRENT_LANG_DIR}
                --language ${LANGUAGE}
        COMMAND
            ${SPHINX_EXECUTABLE} -M ${FORMAT_SPHINX_METHOD}
                ${SPHINX_CURRENT_LANG_DIR} ${SPHINX_BUILD}
                ${CUSTOM_OPTIONS}
                -c ${SPHINX_LATEX_CONF_DIR}
        WORKING_DIRECTORY ${CMAKE_CURRENT_LANG_DIR}
        DEPENDS
            ${DOC_FILES} ${VENV_TARGET} ${SPHINX_TARGET} ${DEPS}
            ${DEPS_STAMP}
        MAIN_DEPENDENCY @{SPHINX_LATEX_CONF_FILE_OUT} ${SPHINX_CONF_FILE_IN}
        COMMENT "Generating ${FORMAT} documentation for ${LANGUAGE} with Sphinx"
    )
    add_custom_target(docs_${LANGUAGE}_${FORMAT} ALL DEPENDS ${SPHINX_TRACKED_FILE})

    if(FORMAT STREQUAL "pdf")
        install(
            FILES
                ${SPHINX_TRACKED_FILE}
            DESTINATION
                ${ARRUS_DOCS_INSTALL_DIR}
            RENAME
                "arrus-${LANGUAGE}.pdf"
            COMPONENT
                docs_${LANGUAGE}_${FORMAT}
        )
    elseif(FORMAT STREQUAL "html")
        install(
            DIRECTORY
                ${SPHINX_TRACKED_DIR}/
            DESTINATION
                ${ARRUS_DOCS_INSTALL_DIR}/html/${LANGUAGE}
            COMPONENT
                docs_${LANGUAGE}_${FORMAT}
        )
    endif()
endfunction()
################################################################################
# Targets and installs.
################################################################################
include(python)
create_python_venv(docs_venv ${CMAKE_CURRENT_BINARY_DIR})
install_sphinx_package(sphinx_package docs_venv)

# C++
## Generate doxygen xml files
find_package(Doxygen)

set(DOXYGEN_INPUT_DIR "${CMAKE_SOURCE_DIR}/arrus/core/api")
file(GLOB_RECURSE DOXYGEN_INPUT_SRC_FILES "${DOXYGEN_INPUT_DIR}/*.h")
string(REPLACE ";" " " DOXYGEN_INPUT_SRC_FILES "${DOXYGEN_INPUT_SRC_FILES}")
set(DOXYGEN_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/cpp/doxy")
set(DOXYGEN_INDEX_FILE ${DOXYGEN_OUTPUT_DIR}/xml/index.xml)

set(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
set(DOXYFILE_OUT ${CMAKE_CURRENT_BINARY_DIR}/cpp/Doxyfile)
configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY)

file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR})
add_custom_command(OUTPUT ${DOXYGEN_INDEX_FILE}
    DEPENDS ${DOXYGEN_INPUT_DIR}
    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
    MAIN_DEPENDENCY ${DOXYFILE_OUT} ${DOXYFILE_IN}
    COMMENT "Generating Doxygen files.")

add_custom_target(arrus_cpp_doxygen ALL DEPENDS ${DOXYGEN_INDEX_FILE})

## C++ sphinx
#CREATE_DOC_TARGET(cpp pdf docs_venv sphinx_package "")
set(ARRUS_BREATHE_PROJECT_OPTION "-Dbreathe_projects.arrus=${DOXYGEN_OUTPUT_DIR}/xml")
CREATE_DOC_TARGET(cpp pdf docs_venv sphinx_package "" ${ARRUS_BREATHE_PROJECT_OPTION})
CREATE_DOC_TARGET(cpp html docs_venv sphinx_package "" ${ARRUS_BREATHE_PROJECT_OPTION})


# Python
if(ARRUS_BUILD_PY)
    install_arrus_package(arrus_package docs_venv python_whl "PKG")
    CREATE_DOC_TARGET(python pdf docs_venv sphinx_package arrus_package "")
    CREATE_DOC_TARGET(python html docs_venv sphinx_package arrus_package "")
endif()

# MATLAB
if(ARRUS_BUILD_MATLAB)
    get_target_property(
            MATLAB_TOOLBOX_DIR
            matlab_toolbox  MATLAB_TOOLBOX_BIN_DIR)
    CREATE_DOC_TARGET(matlab pdf docs_venv sphinx_package matlab_toolbox "")
    CREATE_DOC_TARGET(matlab html docs_venv sphinx_package matlab_toolbox "")
endif()

