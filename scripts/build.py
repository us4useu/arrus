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

    parser.add_argument("--us4r_dir", dest="us4r_dir",
                        type=str, required=False,
                        default=None)

    args = parser.parse_args()
    configuration = args.config
    src_dir = args.source_dir
    us4r_dir = args.us4r_dir

    if src_dir is None:
        raise ValueError("%s environment variable should be declared "
                         "or provided as input parameter."
                         % (SRC_ENVIRON))

    build_dir = os.path.join(src_dir, "build")

    if os.name == "nt":
        pass
    else:
        shell_source(f"{os.path.join(build_dir, 'activate.sh')}")

    cmake_cmd = [
        "cmake",
        "--build", build_dir,
        "--config", configuration
    ]

    print("Calling: %s" % (" ".join(cmake_cmd)))

    current_env = os.environ.copy()
    if us4r_dir is not None:
        current_env["PATH"] = os.path.join(us4r_dir, "lib64") +os.pathsep + current_env["PATH"]

    print(f"Calling with Path {current_env['PATH']}")
    process = subprocess.Popen(cmake_cmd, env=current_env)
    process.wait()
    return_code = process.returncode
    # TODO(pjarosik) consider capturing stderr info and log it into debug
    if return_code != 0:
        raise RuntimeError(f"The process {args} exited with code "
                           f"{return_code}")
    result = subprocess.call(cmake_cmd)

    assert_no_error(result)

def shell_source(script):
    # Credits:
    # https://stackoverflow.com/questions/7040592/calling-the-source-command-from-subprocess-popen#answer-12708396
    pipe = subprocess.Popen(". %s; env" % script, stdout=subprocess.PIPE, shell=True)
    output = pipe.communicate()[0]
    env = dict((line.split("=", 1) for line in str(output).splitlines()))
    os.environ.update(env)

if __name__ == "__main__":
    main()
