import argparse
import os
import subprocess
import re
import shutil
import platform
import requests
import errno
import logging
from datetime import date

COLOR_ERROR = '\033[91m'
COLOR_END = '\033[0m'

SRC_ENVIRON = "ARRUS_SRC_PATH"
INSTALL_ENVIRON = "ARRUS_INSTALL_PATH"

VERSION_TAG_PATTERN = re.compile("^v[0-9\.]+$")


def assert_no_error(return_code):
    if return_code != 0:
        print(COLOR_ERROR+"Failed building targets."+COLOR_END)
        exit(1)


def call_cmd(params):
    print("Calling: %s"%(" ".join(params)))
    return subprocess.call(params)


def get_api_url(repository_name):
    return "https://api.github.com/repos/%s/releases" % repository_name


def get_uploads_url(repository_name):
    return "https://uploads.github.com/repos/%s/releases" % repository_name


def main():
    parser = argparse.ArgumentParser(description="Configures build system.")
    parser.add_argument("--install_dir", dest="install_dir",
                        type=str, required=False,
                        default=os.environ.get(INSTALL_ENVIRON, None))
    parser.add_argument("--repository_name", dest="repository_name", type=str,
                        required=True)
    parser.add_argument("--src_branch_name", dest="src_branch_name", type=str,
                        required=True)
    parser.add_argument("--build_id", dest="build_id", type=str, required=False,
                        default=None)
    parser.add_argument("--token", dest="token", type=str, required=True)

    args = parser.parse_args()
    install_dir = args.install_dir
    repository_name = args.repository_name
    src_branch_name = args.src_branch_name
    token = args.token
    build_id = args.build_id

    if install_dir is None:
        raise ValueError("%s environment variable should be declared "
                         "or provided as input parameter." %(INSTALL_ENVIRON))

    publish(install_dir, token, src_branch_name, repository_name, build_id)


def publish(install_dir, token, src_branch_name, repository_name, build_id):
    timestamp = date.today().strftime("%Y%m%d")
    version = get_version(install_dir)
    if src_branch_name == "master" \
            or VERSION_TAG_PATTERN.match(src_branch_name):
        release_tag = version
        package_name = "arrus-" + release_tag
    elif src_branch_name == "develop":
        release_tag = version + "-dev"
        # Note: the same pattern is used in the root CMakeLists.txt
        package_name = "arrus-" + version + "-dev-" + timestamp
    else:
        raise ValueError("Releases from branch %s are not allowed to be published!")

    package_name += ".zip"
    archive_path = os.path.join(install_dir, f"arrus-{version}.zip")

    release_description = f"date: {timestamp}, tag: {release_tag}, build: {build_id}, host: {platform.node()}"
    response = create_release(repository_name, release_tag, token, release_description)

    if not response.ok:
        resp = response.json()
        if "errors" not in resp:
            print(resp)
            response.raise_for_status()
        if resp["errors"][0]["code"] == "already_exists":
            print("RELEASE EXISTS, OVERWRITING THE RELEASE")
            r = get_release_by_tag(repository_name, release_tag, token)
            r.raise_for_status()
            release_id = r.json()["id"]
            description_body = r.json()["body"]
            release_description = description_body + "\n" + release_description
            r = edit_release(repository_name, release_id, release_tag, token, release_description)
            r.raise_for_status()
        else:
            print(resp)
            response.raise_for_status()
    else:
        release_id = response.json()["id"]

    # Assets.
    r = get_assets(repository_name, release_id, token)
    r.raise_for_status()
    current_assets = r.json()

    existing_assets = [asset for asset in current_assets if asset["name"] == package_name]
    if len(existing_assets) > 0:
        raise RuntimeError(f"Release {release_id} contains more than one asset with name: {package_name}")
    with open(archive_path, "rb") as f:
        data = f.read()
        r = upload_asset(repository_name, release_id, package_name, token, data)
        r.raise_for_status()


def get_assets(repository_name, release_id, token):
    print("Getting assets")
    return requests.get(
        url=get_api_url(repository_name)+"/"+str(release_id)+"/assets",
        headers={
            'Authorization': "token "+token
        }
    )


def delete_asset(repository_name, asset_id, token):
    print("Deleting asset")
    return requests.delete(
        url=get_api_url(repository_name)+"/assets/"+str(asset_id),
        headers={
            'Authorization': "token "+token
        },
    )


def upload_asset(repository_name, release_id, asset_name, token, file_to_upload):
    print("Uploading asset")
    return requests.post(
        url=get_uploads_url(repository_name)+"/"+str(release_id)+"/assets?name="+asset_name,
        headers={
            'Content-Type': 'application/gzip',
            'Authorization': "token "+token
        },
        data=file_to_upload
    )

def get_version(install_dir):
    version_file_path = os.path.join(install_dir, "VERSION.rst")
    with open(os.path.join(version_file_path)) as f:
        version = f.readline()
    if version is None:
        raise ValueError(
            "Invalid version in the version file: %s"%version_file_path)
    return version.strip()


def create_release(repository_name, release, token, body):
    print(f"Creating release: "
          f"repository: {repository_name}, "
          f"release (tag_name): {release} "
          f"body: {body}"
    )
    return requests.post(
        url=get_api_url(repository_name),
        headers={
            'Authorization': "token "+token
        },
        json={
            "tag_name": release,
            "target_commitish": "master",
            "name": release,
            "body": body,
            "draft": False,
            "prerelease": False
        }
    )


def edit_release(repository_name, release_id, release, token, body):
    print("Editing release")
    return requests.patch(
        url=get_api_url(repository_name)+"/"+str(release_id),
        headers={
            'Authorization': "token " + token
        },
        json={
            "tag_name": release,
            "target_commitish": "master",
            "name": release,
            "body": body,
            "draft": False,
            "prerelease": False
        }
    )


def get_release_by_tag(repository_name, release_tag, token):
    print("Getting release by tag")
    return requests.get(
        url=get_api_url(repository_name)+"/tags/"+release_tag,
        headers={
            'Authorization': "token " + token
        }
    )


if __name__ == "__main__":
    main()
