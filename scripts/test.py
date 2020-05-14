import argparse
import os
import subprocess

SRC_ENVIRON = "ARRUS_SRC_PATH"


def assert_no_error(return_code):
    if return_code != 0:
        print("Failed testing targets.")
        exit(1)


def main():
    parser = argparse.ArgumentParser(description="Tests the system.")
    parser.add_argument("--source_dir", dest="source_dir",
                        type=str, required=False,
                        default=os.environ.get(SRC_ENVIRON, None))
    parser.add_argument("--config", dest="config",
                        type=str, required=False, default="Release")

    args = parser.parse_args()
    config = args.config
    src_dir = args.source_dir

    if src_dir is None:
        raise ValueError("%s environment variable should be declared "
                         "or provided as input parameter."
                         % (SRC_ENVIRON))

    build_dir = os.path.join(src_dir, "build")
    os.chdir(build_dir)

    cmake_cmd = [
        "ctest", "-C", config
    ]

    print("Calling: %s" % (" ".join(cmake_cmd)))
    result = subprocess.call(cmake_cmd)
    assert_no_error(result)

if __name__ == "__main__":
    main()
