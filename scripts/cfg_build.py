import argparse
import os
import subprocess
import shutil
import re

SRC_ENVIRON = "ARRUS_SRC_PATH"
US4R_INSTALL_ENVIRON = "US4R_DIR"

COLOR_ERROR = '\033[91m'
COLOR_END = '\033[0m'

VERSION_TAG_PATTERN = re.compile("^v[0-9\.]+$")

def assert_no_error(return_code):
    if return_code != 0:
        print(COLOR_ERROR + "Failed building targets." + COLOR_END)
        exit(1)

def main():
    parser = argparse.ArgumentParser(description="Configures build system.")
    parser.add_argument("--targets", dest="targets",
                        type=str, nargs="*", required=False)
    parser.add_argument("--run_targets", dest="run_targets",
                        type=str, nargs="*", required=False)
    parser.add_argument("--src_branch_name", dest="src_branch_name", type=str,
                        required=False, default="develop")
    parser.add_argument("--source_dir", dest="source_dir",
                        type=str, required=False,
                        default=os.environ.get(SRC_ENVIRON, None))
    parser.add_argument("--us4r_dir", dest="us4r_dir",
                        type=str, default=os.environ.get(US4R_INSTALL_ENVIRON, None))
    parser.add_argument("--options", dest="options",
                        type=str, nargs="*", required=False, default=[])

    args = parser.parse_args()
    targets = args.targets
    run_targets = args.run_targets
    extra_options = args.options
    src_branch_name = args.src_branch_name
    options = []
    if targets is not None:
        options += ["-DARRUS_BUILD_%s=ON" % target.upper() for target in targets]
    if run_targets is not None:
        options += ["-DARRUS_RUN_%s=ON" % t.upper() for t in run_targets]
    options += ["-D%s" % o.upper() for o in extra_options]
    src_dir = args.source_dir
    us4r_install_dir = args.us4r_dir

    if src_dir is None:
        raise ValueError("%s and %s environment variables should be declared or "
                         "provided as a parameters."
                         %(SRC_ENVIRON, US4R_INSTALL_ENVIRON))

    options += ["-DUs4_ROOT_DIR='%s'" % us4r_install_dir]

    if src_branch_name == "master" \
            or VERSION_TAG_PATTERN.match(src_branch_name):
        options += ["-DARRUS_DEVELOP_VERSION=OFF"]

    build_dir = os.path.join(src_dir, "build")

    shutil.rmtree(build_dir, ignore_errors=True)
    os.makedirs(build_dir)

    # Conan install.
    cmd = ["conan", "install",  src_dir, "-if", build_dir]
    result = subprocess.call(cmd)
    assert_no_error(result)

    # Cmake cfg generator.
    if os.name == "nt":
        cmake_generator = "Visual Studio 15 2017 Win64"
    else:
        cmake_generator = "Unix Makefiles"

    cmake_cmd = [
        "cmake",
        "-S", src_dir,
        "-B", build_dir,
        "-G", cmake_generator,
    ] + options
    print("Calling: %s" % (" ".join(cmake_cmd)))
    result = subprocess.call(cmake_cmd)
    assert_no_error(result)


if __name__ == "__main__":
    main()
