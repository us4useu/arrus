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
    parser.add_argument("--source_dir", dest="source_dir",
                        type=str, required=False,
                        default=os.environ.get(SRC_ENVIRON, None))
    parser.add_argument("--install_dir", dest="install_dir",
                        type=str, required=False,
                        default=os.environ.get(INSTALL_ENVIRON, None))

    args = parser.parse_args()
    src_dir = args.source_dir
    install_dir = args.install_dir

    if src_dir is None or install_dir is None:
        raise ValueError("%s and %s environment variables should be declared "
                         "or provided as input parameters."
                         % (SRC_ENVIRON, INSTALL_ENVIRON))

    build_dir = os.path.join(src_dir, "build")

    print("Cleaning up installation dir.")
    shutil.rmtree(install_dir, ignore_errors=True)

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
