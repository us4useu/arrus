import os
import pydevops.cmake as cmake
import pydevops.conan as conan
import pydevops.us4us as us4us


# pydevops version
version = "0.2.0"
# Default branch or tag, which we will be looking for in the
# US4R_API_RELEASE_DIR, if the us4r_api_dir parameters is not provided
# explicitly. Note: the below tag/branch should conform with the us4R required
# version.
us4r_api_default_branch_tag = "v0.11.5"


def get_default_generator_for_current_os():
    if os.name == "nt":
        return "'Visual Studio 15 2017 Win64'"
    else:
        return "'Unix Makefiles'"


def get_default_us4r_api_dir(context):
    if (not context.has_option("us4r_api_dir")
            and not context.has_option("/cfg/cmake/DUs4_ROOT_DIR")):
        if not context.has_option("us4r_api_release_dir"):
            raise ValueError("us4r_api_dir or us4r_api_release_dir must be "
                             "provided.")
        release_dir = context.get_option("us4r_api_release_dir")
        context.remove_option("us4r_api_release_dir")
        return f"{release_dir}/{us4r_api_default_branch_tag}"


stages = {
    "cfg": (
        ("conan", conan.Install),
        ("cmake", cmake.Configure),
    ),
    "build": cmake.Build,
    "test": cmake.Test,
    "install": cmake.Install,
    "package_cpp": us4us.Package,
    "package_matlab": us4us.Package,
    "publish_docs": us4us.PublishDocs,
    "publish_cpp": us4us.PublishReleases,
    "publish_py": us4us.PublishReleases,
    "publish_matlab": us4us.PublishReleases
}

init_stages = ["cfg"]
build_stages = ["build", "test", "install"]

aliases = {
    "us4r_api_dir": "/cfg/cmake/DUs4_ROOT_DIR",
    "build_type": (
        "/cfg/conan/build_type",
        "/cfg/cmake/DCMAKE_BUILD_TYPE",
        "/build/config",
        "/test/C",
        "/install/config"
    ),
    "py": "/cfg/cmake/DARRUS_BUILD_PY",
    "matlab": "/cfg/cmake/DARRUS_BUILD_MATLAB",
    "docs": "/cfg/cmake/DARRUS_BUILD_DOCS",
    "tests": "/cfg/cmake/DARRUS_RUN_TESTS",
    "j": "/build/j"
}

defaults = {
    "build_type": "Release",
    "us4r_api_dir": get_default_us4r_api_dir,
    "/cfg/cmake/generator": get_default_generator_for_current_os(),
    "/cfg/cmake/DARRUS_EMBED_DEPS": "ON",
    "/install/prefix": "./install"
}
