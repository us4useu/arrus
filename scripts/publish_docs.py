import argparse
import os
import subprocess
import shutil
import platform

COLOR_ERROR = '\033[91m'
COLOR_END = '\033[0m'

SRC_ENVIRON = "ARRUS_SRC_PATH"
INSTALL_ENVIRON = "ARRUS_INSTALL_PATH"


def assert_no_error(return_code):
    if return_code != 0:
        print(COLOR_ERROR + "Failed building targets." + COLOR_END)
        exit(1)


def call_cmd(params):
    print("Calling: %s" % (" ".join(params)))
    return subprocess.call(params)

def git_commit(msg):
    params = ["git", "commit", "-m", "'"+msg+"'"]
    print("Calling: %s"%(" ".join(params)))
    try:
        out = subprocess.check_output(params)
    except subprocess.CalledProcessError as e:
        out = str(e.output)
        print(out)
        if "nothing to commit" in out:
            return "ntc"
        else:
            return "fail"
    print("Commit output: %s" % out)
    return "ok"


def main():
    parser = argparse.ArgumentParser(description="Configures build system.")
    parser.add_argument("--install_dir", dest="install_dir",
                        type=str, required=False,
                        default=os.environ.get(INSTALL_ENVIRON, None))
    parser.add_argument("--repository", dest="repository", type=str, required=True)
    parser.add_argument("--src_branch_name", dest="src_branch_name", type=str, required=True)
    parser.add_argument("--build_id", dest="build_id", type=str, required=False, default=None)

    args = parser.parse_args()
    install_dir = args.install_dir
    repository = args.repository
    src_branch_name = args.src_branch_name
    build_id = args.build_id

    if install_dir is None:
        raise ValueError("%s environment variable should be declared "
                         "or provided as input parameter."
                         % (INSTALL_ENVIRON))

    # Get repository name.
    _, repository_name = os.path.split(repository)
    repository_name, _ = repository_name.split(".")

    try:
        publish(build_id, install_dir, repository, repository_name,
                src_branch_name)
    finally:
        os.chdir("..")
        print("Cleaning up")
        print(os.getcwd())
        shutil.rmtree(
            repository_name)


def publish(build_id, install_dir, repository, repository_name,
            src_branch_name):
    # Cleanup if necessary.
    shutil.rmtree(
        repository_name,
        ignore_errors=True)
    result = call_cmd(["git", "clone", repository])
    assert_no_error(result)
    # Get version number
    if src_branch_name == "master":
        # Generate specific version and 'current' releases.
        version_file_path = os.path.join(install_dir, "VERSION.rst")
        version = None
        with open(os.path.join(version_file_path)) as f:
            version = f.readline()
        if version is None:
            raise ValueError(
                "Invalid version in the version file: %s"%version_file_path
            )
        releases = [version, "current"]
        pass
    elif src_branch_name == "develop":
        releases = ["develop"]
    else:
        raise ValueError("Docs from branch %s are not allowed to be published!")
    for release in releases:
        release_dir = os.path.join(repository_name, "releases", release)
        docs_dir = os.path.join(install_dir, "docs", "html")
        shutil.rmtree(
            release_dir,
            ignore_errors=True)
        os.makedirs(release_dir)
        language_doc_dirs = os.listdir(docs_dir)
        for d in language_doc_dirs:
            dst = os.path.join(docs_dir, d)
            src = os.path.join(release_dir, d)
            shutil.copytree(dst, src)
    os.chdir(repository_name)
    result = call_cmd(["git", "add", "-A"])
    assert_no_error(result)
    commit_msg = "Updated docs"
    if build_id is not None:
        hostname = platform.node()
        commit_msg += ", build: %s, host: %s"%(build_id, hostname)
    result = git_commit(commit_msg)
    if result == 'ntc':
        print("Nothing to commit")
        return
    elif result != "ok":
        raise ValueError("Something wrong when commiting the changes.")
    result = call_cmd(["git", "push", repository])
    assert_no_error(result)

if __name__ == "__main__":
    main()
