#     ██████████      ███████                     
#    ▒▒███▒▒▒▒███   ███▒▒▒▒▒███                   
#     ▒███   ▒▒███ ███     ▒▒███                  
#     ▒███    ▒███▒███      ▒███                  
#     ▒███    ▒███▒███      ▒███                  
#     ▒███    ███ ▒▒███     ███                   
#     ██████████   ▒▒▒███████▒                    
#    ▒▒▒▒▒▒▒▒▒▒      ▒▒▒▒▒▒▒                      
#                                                                                
#                                                 
#     ██████   █████    ███████    ███████████    
#    ▒▒██████ ▒▒███   ███▒▒▒▒▒███ ▒█▒▒▒███▒▒▒█    
#     ▒███▒███ ▒███  ███     ▒▒███▒   ▒███  ▒     
#     ▒███▒▒███▒███ ▒███      ▒███    ▒███        
#     ▒███ ▒▒██████ ▒███      ▒███    ▒███        
#     ▒███  ▒▒█████ ▒▒███     ███     ▒███        
#     █████  ▒▒█████ ▒▒▒███████▒      █████       
#    ▒▒▒▒▒    ▒▒▒▒▒    ▒▒▒▒▒▒▒       ▒▒▒▒▒        
#                                                 
#                                                 
#      █████████  █████   █████ █████ ███████████ 
#     ███▒▒▒▒▒███▒▒███   ▒▒███ ▒▒███ ▒▒███▒▒▒▒▒███
#    ▒███    ▒▒▒  ▒███    ▒███  ▒███  ▒███    ▒███
#    ▒▒█████████  ▒███████████  ▒███  ▒██████████ 
#     ▒▒▒▒▒▒▒▒███ ▒███▒▒▒▒▒███  ▒███  ▒███▒▒▒▒▒▒  
#     ███    ▒███ ▒███    ▒███  ▒███  ▒███        
#    ▒▒█████████  █████   █████ █████ █████       
#     ▒▒▒▒▒▒▒▒▒  ▒▒▒▒▒   ▒▒▒▒▒ ▒▒▒▒▒ ▒▒▒▒▒        
#
# Reason: std4us pointing at a development branch

import os
import pydevops.cmake as cmake
import pydevops.conan as conan
import pydevops.gitdep as gitdep
import pydevops.us4us as us4us
import pydevops.utils as utils

def get_generator_options_for_current_os():
    if os.name == "nt":
        if utils.is_version_at_least("0.5.0"):
            return {
                # Use Visual Studio's ClangCL
                # In VS2026, the version is Clang 21
                "toolset": "ClangCL",
                "generator": "'Visual Studio 18 2026'",
            }
        else:
            print("Consider upgrading pydevops to version 0.5.0 or later for better compatibility with Visual Studio 2017 and later.")
            return {
                "generator": "'Visual Studio 15 2017 Win64'"
            }
    else:
        return {
            "generator": "'Unix Makefiles'"
        }


# pydevops version
version = "0.2.0"
# Default branch or tag, which we will be looking for in the
# US4R_API_RELEASE_DIR, if the us4r_api_dir parameters is not provided
# explicitly. Note: the below tag/branch should conform with the us4R required
# version.
us4r_api_default_branch_tag = "v0.15.0"

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
        ("fetch_std4us", gitdep.Fetch),
        ("add_std4us_index", conan.AddLocalIndex),
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
    "/cfg/cmake/DARRUS_EMBED_DEPS": "ON",
    "/install/prefix": "./install",
    "/cfg/fetch_std4us/repo": "https://github.com/us4useu/std4us.git",
    "/cfg/fetch_std4us/revision": "clang-dev",
    "/cfg/fetch_std4us/path": "std4us",
    "/cfg/add_std4us_index/path": "std4us",
}

defaults.update(get_generator_options_for_current_os())

transforms = [
    lambda options: {
        f"/cfg/cmake/preset": f"conan-{options['/build/config'].lower()}", 
        f"/build/preset": f"conan-{options['/build/config'].lower()}",
        f"/test/preset": f"conan-{options['/build/config'].lower()}",
        },
    lambda options: {
        f"/install/build_dir_suffix": f"/{options['/build/config']}/"
    },
]