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
    parser.add_argument("--source_dir", dest="source_dir",
                        type=str, required=False,
                        default=os.environ.get(SRC_ENVIRON, None))

    args = parser.parse_args()
    configuration = args.config
    src_dir = args.source_dir

    if src_dir is None:
        raise ValueError("%s environment variable should be declared "
                         "or provided as input parameter."
                         % (SRC_ENVIRON))

    build_dir = os.path.join(src_dir, "build")

    cmake_cmd = [
        "cmake",
        "--build", build_dir,
        "--config", configuration
    ]

    print("Calling: %s" % (" ".join(cmake_cmd)))
    result = subprocess.call(cmake_cmd)
    assert_no_error(result)

if __name__ == "__main__":
    main()
