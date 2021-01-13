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
    parser.add_argument("--j", dest="j", type=int, required=False,
                        default=1)
    parser.add_argument("--verbose", dest="verbose",
                        required=False, default=False,
                        action="store_true")

    args = parser.parse_args()
    configuration = args.config
    src_dir = args.source_dir
    us4r_dir = args.us4r_dir
    verbose = args.verbose
    number_of_jobs = args.j

    if src_dir is None:
        raise ValueError("%s environment variable should be declared "
                         "or provided as input parameter."
                         % (SRC_ENVIRON))

    build_dir = os.path.join(src_dir, "build")

    cmake_cmd = []
    join_cmd = False
    if os.name == "nt":
        join_cmd = True
        cmd = os.path.join(build_dir, 'activate.bat')
        cmake_cmd += [f'"{cmd}"', "&&"]
        pass
    else:
        join_cmd = False
        shell_source(f"{os.path.join(build_dir, 'activate.sh')}")

    build_dir = os.path.abspath(build_dir)
    if os.name == "nt":
        # Properly handle paths with white spaces.
        build_dir = f'"{build_dir}"'
    cmake_cmd += [
        "cmake",
        "--build", build_dir,
        "--config", configuration,
    ]

    if os.name != "nt":
        cmake_cmd += [
            "-j" + str(number_of_jobs)
        ]

    if verbose:
        cmake_cmd += ["--verbose"]

    print("Calling: %s" % (" ".join(cmake_cmd)))

    current_env = os.environ.copy()
    if us4r_dir is not None:
        current_env["PATH"] = os.path.join(us4r_dir, "lib64") +os.pathsep + current_env["PATH"]

    print(f"Calling with Path {current_env['PATH']}")
    if join_cmd:
        cmake_cmd = " ".join(cmake_cmd)
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
    env = (line.decode("utf-8") for line in output.splitlines())
    env = (line.split("=", 1) for line in env)
    env = (var for var in env if len(var) == 2) # // leave correct pairs only
    env = dict(env)
    os.environ.update(env)

if __name__ == "__main__":
    main()
