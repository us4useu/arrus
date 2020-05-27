import argparse
import os
import subprocess
import shutil
import tempfile
import urllib.request
import re
import errno

COLOR_ERROR = '\033[91m'
COLOR_END = '\033[0m'

SRC_ENVIRON = "ARRUS_SRC_PATH"
INSTALL_ENVIRON = "ARRUS_INSTALL_PATH"


def get_firmware_tag_name(firmware_version, tx_firmware_version):
    return f"us4oem-firmware-{firmware_version}{tx_firmware_version}"


def get_firmware_url_pattern(firmware_version, tx_firmware_version):
    name = get_firmware_tag_name(firmware_version, tx_firmware_version)
    return f"https://github.com/us4useu/arrus-public/releases/download/{name}/{name}.zip"

def _search_for_single_pattern(pattern, content):
    for line in content:
        result = re.findall(pattern, line)
        if len(result) == 1:
            return result[0]


def get_required_firmware_version(install_dir):
    with open(os.path.join(install_dir, "Version.rst")) as f:
        content = f.readlines()
    arrus_version = _search_for_single_pattern(
        r"^\s*arrus:\s+([0-9\.]+)\s*$",
        content)
    # Search for firmware version.
    firmware = _search_for_single_pattern(r"^\s*firmware:\s+([a-fA-F0-9]+)\s*$",
                                          content)
    print(f"Firmware: {firmware}")
    # Search for tx firmware version.
    tx_firmware = _search_for_single_pattern(
        r"^\s*tx\s+firmware:\s+([a-fA-F0-9]+)\s*$", content)
    print(f"tx firmware: {tx_firmware}")
    return arrus_version, firmware, tx_firmware

def assert_no_error(return_code):
    if return_code != 0:
        print(COLOR_ERROR + "Failed building targets." + COLOR_END)
        exit(1)


def main():
    parser = argparse.ArgumentParser(description=
                                     "Creates the final package which will be "
                                     "provided to user.")
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

    arrus_version, firmware_version, tx_firmware_version = \
        get_required_firmware_version(install_dir)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Zip the source file and place it in the installer directory
        install_py_path = os.path.join(src_dir, "build/installer/install.py")
        dist_path = tmp_dir
        subprocess.call(["pyinstaller", install_py_path, "--distpath",
                         dist_path])

        # The name of destination directory is determined by pyinstaller.
        destination_path = os.path.join(tmp_dir, "install")
        shutil.make_archive(os.path.join(destination_path, "arrus"),
                            "zip", install_dir)
        firmware_url = get_firmware_url_pattern(firmware_version,
                                                tx_firmware_version)
        output_file = get_firmware_tag_name(firmware_version,
                                            tx_firmware_version) + ".zip"
        output_file_abs_path = os.path.join(destination_path, output_file)

        print(f"Downloading {output_file}, url: {firmware_url}")
        urllib.request.urlretrieve(firmware_url, output_file_abs_path)
        print("Downloading finished.")
        destination_archive = os.path.join(install_dir, f"arrus-{arrus_version}")
        try:
            os.remove(destination_archive + ".zip")
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise e
        shutil.make_archive(destination_archive, "zip", destination_path)
        print(f"Created file {destination_archive}")



if __name__ == "__main__":
    main()
