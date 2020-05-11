import argparse
import os
import subprocess
import shutil

SRC_ENVIRON = "ARRUS_SRC_PATH"
INSTALL_ENVIRON = "ARRUS_INSTALL_PATH"

COLOR_ERROR = '\033[91m'
COLOR_END = '\033[0m'

def assert_no_error(return_code):
    if return_code != 0:
        print(COLOR_ERROR + "Failed building targets." + COLOR_END)
        exit(1)

def main():
    parser = argparse.ArgumentParser(description="Configures build system.")
    parser.add_argument("--targets", dest="targets",
                        type=str, nargs="+", required=True)
    parser.add_argument("--source_dir", dest="source_dir",
                        type=str, required=False,
                        default=os.environ.get(SRC_ENVIRON, None))
    parser.add_argument("--install_dir", dest="install_dir",
                        type=str, required=False,
                        default=os.environ.get(INSTALL_ENVIRON, None))

    args = parser.parse_args()
    targets = args.targets
    options = ["-DARRUS_BUILD_%s=ON" % target.upper() for target in targets]
    src_dir = args.source_dir
    install_dir = args.install_dir

    if src_dir is None or install_dir is None:
        raise ValueError("%s and %s environment variables should be declared"
                         % (SRC_ENVIRON, INSTALL_ENVIRON))
    options += ["-DUs4_ROOT_DIR=" + install_dir]

    build_dir = os.path.join(src_dir, "build")

    shutil.rmtree(build_dir, ignore_errors=True)
    os.makedirs(build_dir)

    cmake_generator = ""
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
