import argparse
import os
import subprocess
import shutil

COLOR_ERROR = '\033[91m'
COLOR_END = '\033[0m'

SRC_ENVIRON = "ARRUS_SRC_PATH"
INSTALL_ENVIRON = "ARRUS_INSTALL_PATH"

def assert_no_error(return_code):
    if return_code != 0:
        print(COLOR_ERROR + "Failed building targets." + COLOR_END)
        exit(1)

def main():
    parser = argparse.ArgumentParser(description="Configures build system.")
    parser.add_argument("--config", dest="config",
                        type=str, required=False, default="Release")
    parser.add_argument("--targets", dest="targets",
                        type=str, required=False, nargs="*")

    args = parser.parse_args()
    targets = args.targets
    configuration = args.config

    src_dir = os.environ.get(SRC_ENVIRON, None)
    install_dir = os.environ.get(INSTALL_ENVIRON, None)
    if src_dir is None or install_dir is None:
        raise ValueError("%s and %s environment variables should be declared"
                         % (SRC_ENVIRON, INSTALL_ENVIRON))

    build_dir = os.path.join(src_dir, "build")

    extra_options = []
    if args.targets is not None:
        extra_options = [
            "--target", " ".join(args.targets)
        ]

    cmake_cmd = [
        "cmake",
        "--build", build_dir,
        "--config", configuration
    ] + extra_options

    print("Calling: %s" % (" ".join(cmake_cmd)))
    result = subprocess.call(cmake_cmd)
    assert_no_error(result)

    cmake_install_cmd = [
        "cmake",
        "--install", build_dir,
        "--prefix", install_dir
    ]
    print("Calling: %s"%(" ".join(cmake_install_cmd)))
    result = subprocess.call(cmake_install_cmd)
    assert_no_error(result)



if __name__ == "__main__":
    main()
