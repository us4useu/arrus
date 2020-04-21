################################################################################
# Sphinx
################################################################################
set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR})

set(DOC_FILES
    "index.rst"
    "content/api.python.rst"
    "content/api.matlab.rst"
    "content/release_notes.rst"
    "content/examples/index.rst"
    "content/examples/single_module.python.rst"
    "content/examples/single_module.matlab.rst"
)
list(TRANSFORM DOC_FILES PREPEND "${SPHINX_SOURCE}/")

################################################################################
# Sphinx Docs generator.
################################################################################
function(CREATE_DOC_TARGET LANGUAGE FORMAT VENV_TARGET SPHINX_TARGET DEPS)
    get_target_property(VENV_PY_EXECUTABLE ${VENV_TARGET} VENV_EXECUTABLE)
    get_target_property(SPHINX_EXECUTABLE ${SPHINX_TARGET} SPHINX_EXECUTABLE)
    if(NOT "${DEPS}" STREQUAL "")
        get_target_property(ARRUS_INSTALL_STAMP ${DEPS} INSTALL_TIMESTAMP)
    endif()
    if(FORMAT STREQUAL "pdf")
        set(FORMAT_OUT_DIR "latex")
        set(FORMAT_SPHINX_METHOD "latexpdf")
        set(FORMAT_TRACKED_FILE "arius-sdk${LANGUAGE}.pdf")
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
                -c ${SPHINX_LATEX_CONF_DIR}
        WORKING_DIRECTORY ${CMAKE_CURRENT_LANG_DIR}
        DEPENDS
            ${DOC_FILES} ${VENV_TARGET} ${SPHINX_TARGET} ${DEPS}
            ${ARIUS_INSTALL_STAMP}
        MAIN_DEPENDENCY @{SPHINX_LATEX_CONF_FILE_OUT} ${SPHINX_CONF_FILE_IN}
        COMMENT "Generating ${FORMAT} documentation for ${LANGUAGE} with Sphinx"
    )
    add_custom_target(docs_${LANGUAGE}_${FORMAT} ALL DEPENDS ${SPHINX_TRACKED_FILE})

    if(FORMAT STREQUAL "pdf")
        install(
            FILES
                ${SPHINX_TRACKED_FILE}
            DESTINATION
                ${ARIUS_DOCS_INSTALL_DIR}
            RENAME
                "arius-sdk-${LANGUAGE}.pdf"
            COMPONENT
                docs_${LANGUAGE}_${FORMAT}
        )
    elseif(FORMAT STREQUAL "html")
        install(
            DIRECTORY
                ${SPHINX_TRACKED_DIR}/
            DESTINATION
                ${ARIUS_DOCS_INSTALL_DIR}/html/${LANGUAGE}
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

if(ARIUS_BUILD_PY)
    install_arius_package(arius_package docs_venv python_whl)
    CREATE_DOC_TARGET(python pdf docs_venv sphinx_package arius_package)
    CREATE_DOC_TARGET(python html docs_venv sphinx_package arius_package)
endif()

if(ARIUS_BUILD_MATLAB)
    CREATE_DOC_TARGET(matlab pdf docs_venv sphinx_package "")
    CREATE_DOC_TARGET(matlab html docs_venv sphinx_package "")
endif()

