################################################################################
# Sphinx
################################################################################
find_package(Sphinx REQUIRED)

set(SPHINX_SOURCE .)
set(SPHINX_BUILD ${CMAKE_CURRENT_BINARY_DIR}/sphinx)

set(DOC_FILES
    "index.rst"
    "content/api.rst"
    "content/release_notes.rst"
    "content/user/examples/index.rst"
)
list(TRANSFORM DOC_FILES PREPEND "${SPHINX_SOURCE}/")
file(MAKE_DIRECTORY ${SPHINX_BUILD})

################################################################################
# Sphinx Latex PDF geneartor
################################################################################
set(SPHINX_LATEX_CONF_FILE_IN conf.py.in)
set(SPHINX_LATEX_CONF_DIR ${CMAKE_CURRENT_BINARY_DIR}/latex)
set(SPHINX_LATEX_CONF_FILE_OUT ${SPHINX_LATEX_CONF_DIR}/conf.py)
configure_file(${SPHINX_LATEX_CONF_FILE_IN} ${SPHINX_LATEX_CONF_FILE_OUT} @ONLY)

set(SPHINX_PDF_FILE "${SPHINX_BUILD}/latex/arius-sdk-python.pdf")

add_custom_command(OUTPUT ${SPHINX_PDF_FILE}
    COMMAND
        ${SPHINX_EXECUTABLE} -M latexpdf
        ${SPHINX_SOURCE} ${SPHINX_BUILD}
        -c ${SPHINX_LATEX_CONF_DIR}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS
        ${DOC_FILES}
    MAIN_DEPENDENCY @{SPHINX_LATEX_CONF_FILE_OUT} ${SPHINX_LATEX_CONF_FILE_IN}
    COMMENT "Generating PDF documentation with Sphinx"
)

add_custom_target(python_sphinx_pdf ALL DEPENDS ${SPHINX_PDF_FILE})

install(
    FILES
        ${SPHINX_PDF_FILE}
    DESTINATION
        ${ARIUS_DOCS_INSTALL_DIR}
)

