import argparse
import os
import subprocess
import shutil

SRC_ENVIRON = "ARRUS_SRC_PATH"
INSTALL_ENVIRON = "ARRUS_INSTALL_PATH"

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
            "--target", args.targets
        ]

    cmake_cmd = [
        "cmake",
        "--build", build_dir,
        "--config", configuration
    ] + extra_options

    print("Calling: %s" % (" ".join(cmake_cmd)))
    subprocess.call(cmake_cmd)

    cmake_install_cmd = [
        "cmake",
        "--install", build_dir,
        "--prefix", install_dir
    ]
    print("Calling: %s"%(" ".join(cmake_install_cmd)))
    subprocess.call(cmake_install_cmd)



if __name__ == "__main__":
    main()
