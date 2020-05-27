import abc
import logging
from logging import INFO, DEBUG, WARNING, ERROR
import colorama
import os
import argparse
import shutil
import zipfile
import winreg
import subprocess
import tempfile
import yaml
import ctypes
from pathlib import Path

PROJECT_VERSION = "${PROJECT_VERSION}"
FIRMWARE_VERSION = "${Us4OEM_FIRMWARE_VERSION}"
TX_FIRMWARE_VERSION = "${Us4OEM_TX_FIRMWARE_VERSION}"

# Logging
_logger = logging.getLogger("arrus installer")
_logger.setLevel(logging.DEBUG)
LOGGING_FILE = "installer.log"
# Console output
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter(
    "%(message)s")
console_handler.setFormatter(console_formatter)
# File output
log_file_handler = logging.FileHandler(LOGGING_FILE)
log_file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log_file_handler.setFormatter(file_formatter)
_logger.addHandler(console_handler)
_logger.addHandler(log_file_handler)

# Constants:

_EXPECTED_LIB_DIR = "lib64"
_EXPECTED_LIBS = ["Us4OEM.dll"]
_PKG_ZIP = "arrus.zip"
_FIRMWARE_ZIP = f"us4oem-firmware-{FIRMWARE_VERSION}{TX_FIRMWARE_VERSION}.zip"
_FIRMWARE_RPD_FILE = f"us4OEM_rev_{FIRMWARE_VERSION}.rpd"
# TODO(pjarosik) make sure, that sea and sed files will always be in this format
_FIRMWARE_SEA_FILE = f"us4OEM_tx_rev_{FIRMWARE_VERSION[0].upper()}.sea"
_FIRMWARE_SED_FILE = f"us4OEM_tx_rev_{FIRMWARE_VERSION[0].upper()}.sed"

_US4OEM_STATUS_BIN = "bin/us4OEMStatus"
_US4OEM_FIRMWARE_UPDATE_BIN = "bin/us4OEMFirmwareUpdate"

_STATUS_US4OEM_KEY = "us4oems"


class InstallationContext:
    # tempfile.TemporaryDirectory()
    workspace_dir: object = None
    existing_install_dir: str = None
    install_dir: str = None
    abort: bool = False
    override_existing: bool = False
    system_status: dict = None

    def cleanup(self):
        if self.workspace_dir is not None:
            self.workspace_dir.cleanup()


class Stage(abc.ABC):

    @abc.abstractmethod
    def read_context_from_params(self, args, ctx: InstallationContext) -> bool:
        pass

    @abc.abstractmethod
    def ask_user_for_context(self, ctx: InstallationContext) -> bool:
        pass

    @abc.abstractmethod
    def process(self, context: InstallationContext) -> bool:
        """
        Should return True, if the installation should continue.
        """
        pass

    def ask_ynq(self, msg: str, ctx: InstallationContext) -> bool:
        while True:
            answer = input(f"{msg} [(Y)es/(N)o/(Q)uit]: ").lower().strip()
            if answer in {"yes", "y"}:
                return True
            elif answer in {"no", "n"}:
                return False
            elif answer in {"quit", "q"}:
                ctx.abort = True
                return None
            else:
                print("Please answer (Y)es/(N)o/(Q)uit")

    def ask_yn(self, msg: str, ctx: InstallationContext) -> bool:
        while True:
            answer = input(f"{msg} [(Y)es/(N)o]: ").lower().strip()
            if answer in {"yes", "y"}:
                return True
            elif answer in {"no", "n"}:
                return False
            else:
                print("Please answer (Y)es or (N)o")

    def ask_for_path(self, msg: str, default, ctx: InstallationContext) -> str:
        while True:
            answer = input(f"{msg} [press enter for default: {default}]: ").strip()
            if not answer:
                return default
            else:
                return answer

    def run_subprocess(self, args):
        try:
            subprocess.check_output(args)
        except subprocess.CalledProcessError as e:
            _logger.log(ERROR, f"Error calling {args}. "
                               f"Return code: {e.returncode}, "
                               f"output: {e.output}")
            raise e


class WelcomeStage(Stage):
    """
    Welcome stage:
    - show version of the installed package
    - check if user runs program as administrator
    - ask to confirm installation
    - creates temporary directory for any files that will be used in the system
      ("workspace")
    """

    def read_context_from_params(self, args, ctx: InstallationContext) -> bool:
        return True

    def ask_user_for_context(self, ctx: InstallationContext) -> bool:
        return True

    def process(self, context: InstallationContext) -> bool:
        print(f"Starting ARRUS {PROJECT_VERSION} installer...")
        # Check if current user is administrator.
        if ctypes.windll.shell32.IsUserAnAdmin() == 0:
            _logger.log(ERROR, f"Run this program as an administrator.")
            return False
        # Confirm to install software
        ans = self.ask_yn(f"Software installation and firmware update may take "
                    f"several hours. Are you sure you want to continue?",
                    ctx=context)
        context.workspace_dir = tempfile.TemporaryDirectory()
        return ans


class FindExistingInstallationStage(Stage):

    def read_context_from_params(self, args, ctx: InstallationContext) -> bool:
        return True

    def ask_user_for_context(self, ctx: InstallationContext) -> bool:
        return True

    def process(self, context: InstallationContext) -> bool:
        _logger.log(DEBUG, "Checking for existing installations...")
        existing_installations = self.find_arrus_paths()

        if len(existing_installations) == 0:
            _logger.log(DEBUG, "There is no existing installation on this "
                               "computer.")
            context.existing_install_dir = None

        elif len(existing_installations) > 1:
            _logger.log(WARNING, "Found multiple ARRUS installations on this "
                                 "computer: %s" %
                        (",".join(existing_installations)))
            # We use the first one from PATH variable - the one currently
            # used.
            context.existing_install_dir = existing_installations[0]
        else:
            _logger.log(DEBUG, "Found exactly one installation on this "
                               "computer: %s." % existing_installations[0])
            context.existing_install_dir = existing_installations[0]

        return True

    @staticmethod
    def find_arrus_paths():
        result = []
        paths = os.environ["PATH"].split(os.pathsep)
        for path in paths:
            for expected_lib in _EXPECTED_LIBS:
                full_path = os.path.join(path, expected_lib)
                lib_file = Path(full_path)
                if lib_file.is_file():
                    root_dir, lib_dir = os.path.split(path)
                    if lib_dir == _EXPECTED_LIB_DIR:
                        result.append(root_dir)
        return result


class UnzipFilesStage(Stage):
    UNZIP_DEFAULT_DIR = "C:\\local\\arrus"

    def read_context_from_params(self, args, ctx: InstallationContext):
        pass

    def ask_user_for_context(self, ctx: InstallationContext):
        path = None
        if ctx.existing_install_dir is not None:
            msg = f"Found ARRUS in path '{ctx.existing_install_dir}'.\n" \
                  f"Would you like to replace it with newer version?"
            answer = self.ask_ynq(msg, ctx)
            if answer is None or ctx.abort:
                return False
            if answer:
                path = ctx.existing_install_dir
        if path is None:
            path = self.ask_for_path(
                "Please provide destination path",
                default=UnzipFilesStage.UNZIP_DEFAULT_DIR,
                ctx=ctx
            )
        if os.path.exists(path):
            if os.path.isfile(path):
                msg = f"WARNING: The FILE '{path}' will be deleted! "
            elif os.path.isdir(path):
                msg = f"WARNING: The contents of '{path}' will be deleted! "
            else:
                raise ValueError(f"Unrecognized file system object: {path}")
            print(f"{colorama.Fore.LIGHTRED_EX}{msg}{colorama.Style.RESET_ALL}")
            result = self.ask_yn("Are you sure you want to continue?", ctx)
            if not result:
                ctx.abort = True
                return False
            else:
                ctx.override_existing = True
        ctx.install_dir = path
        return True

    def process(self, ctx: InstallationContext):
        if os.path.exists(ctx.install_dir):
            if ctx.override_existing:
                if os.path.isdir(ctx.install_dir):
                    shutil.rmtree(ctx.install_dir)
                elif os.path.isfile(ctx.install_dir) \
                  or os.path.islink(ctx.install_dir):
                    os.remove(ctx.install_dir)
            else:
                raise ValueError("Overriding existing directories or files is "
                                 "forbidden!")
        with zipfile.ZipFile(_PKG_ZIP, "r") as zfile:
            zfile.extractall(ctx.install_dir)
        return True


class UpdateEnvVariablesStage(Stage):

    def read_context_from_params(self, args, ctx: InstallationContext):
        pass

    def ask_user_for_context(self, ctx: InstallationContext):
        pass

    def process(self, ctx: InstallationContext):
        reg = winreg.ConnectRegistry(None, winreg.HKEY_CURRENT_USER)
        key = winreg.OpenKey(reg,
            r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment")
        current_system_path, d_type = winreg.QueryValueEx(key, "Path")

        install_path = os.path.join(
            ctx.install_dir,
            _EXPECTED_LIB_DIR
        )

        system_paths = current_system_path.split(os.pathsep)
        UpdateEnvVariablesStage.update_paths_list(system_paths, install_path)
        subprocess.check_output(["setx", "Path", system_paths, "/M"])
        return True

    @staticmethod
    def update_paths_list(paths_list, install_path):
        lib_path = Path(os.path.join(install_path, _EXPECTED_LIB_DIR))
        us4oem_paths = (os.path.join(p, _EXPECTED_LIBS[0]) for p in paths_list)
        us4oem_paths = ((i, Path(p)) for i, p in enumerate(us4oem_paths))
        us4oem_paths = ((i, p) for i, p in us4oem_paths if p.is_file())
        current_us4oem_path = next(us4oem_paths, None)
        if current_us4oem_path is not None:
            index, path = current_us4oem_path
            if lib_path.resolve() == path.parent.resolve():
                return
        else:
            index = len(paths_list)
        paths_list.insert(index, lib_path.resolve())


class UpdateFirmwareStage(Stage):
    """
    Checks the status of connected devices.
    - get all modules available in the system
    - get theirs firmware version (without Tx firmware version - tx firmware
      may not be initializable currently!)
    - if there is no device - exit with error message

    Update main firmware of each module if necessary.
    """

    def read_context_from_params(self, args, ctx: InstallationContext) -> bool:
        raise NotImplementedError

    def ask_user_for_context(self, ctx: InstallationContext) -> bool:
        return True

    def process(self, context: InstallationContext) -> bool:
        # Get status of the available modules.
        _logger.log(INFO, "Checking the status of available Us4OEMs")
        modules_key = _STATUS_US4OEM_KEY
        status = self.get_status_yaml(context)
        if status is None \
                or modules_key not in status \
                or len(status[modules_key]) == 0:
            _logger.log(ERROR, "No available Us4OEM module has been detected. "
                               "Make sure that the device is properly connected.")
            context.abort = True
            return False
        context.system_status = status

        # Run main firmware update if necessary.
        for module in context.system_status["us4oems"]:
            module_tx_firmware_version = module["firmwareVersion"].strip()
            if module_tx_firmware_version != FIRMWARE_VERSION.strip():
                module_index = module['index']
                self.run_us4oem_firmware_update(context, module_index)

        context.system_status = self.get_status_yaml(context, tx_firmware=True)

        for module in context.system_status["us4oems"]:
            module_tx_firmware_version = module["txFirmwareVersion"].strip()
            if module_tx_firmware_version != TX_FIRMWARE_VERSION.strip():
                module_index = module['index']
                self.run_us4oem_tx_firmware_update(context, module_index)

    def get_status_yaml(self, context: InstallationContext, tx_firmware=False):
        # Run us4oemStatus
        binary = os.path.join(context.install_dir, _US4OEM_STATUS_BIN)
        output_file = os.path.join(context.workspace_dir.name, "status.yml")

        to_run = [binary, "--output-file", output_file]
        if tx_firmware:
            to_run += ["--tx-firmware-version"]
        self.run_subprocess(to_run)
        # Read the yaml file
        with open(output_file, "r") as f:
            return yaml.safe_load(f)

    def run_us4oem_firmware_update(self, context, module_index):
        update_bin = os.path.join(context.install_dir,
                                            _US4OEM_FIRMWARE_UPDATE_BIN)
        firmware_dir = self.unzip_firmware_if_necessary(context)

        rpd_file_path = os.path.join(firmware_dir.name, _FIRMWARE_RPD_FILE)

        self.run_subprocess([update_bin,
                             "--rpd-file", rpd_file_path,
                             "--us4OEM-indices", module_index])
        pass

    def run_us4oem_tx_firmware_update(self, context, module_index):
        update_bin = os.path.join(context.install_dir,
                                            _US4OEM_FIRMWARE_UPDATE_BIN)
        firmware_dir = self.unzip_firmware_if_necessary(context)

        sea_file_path = os.path.join(firmware_dir.name, _FIRMWARE_SEA_FILE)
        sed_file_path = os.path.join(firmware_dir.name, _FIRMWARE_SED_FILE)

        self.run_subprocess([update_bin,
                             "--sea-file", sea_file_path,
                             "--sed-file", sed_file_path,
                             "--us4OEM-indices", module_index])

    def unzip_firmware_if_necessary(self, context: InstallationContext):
        firmware_dir = Path(os.path.join(context.workspace_dir.name, "firmware"))
        if not firmware_dir.is_dir():
            firmware_dir.mkdir(parents=True)
            with zipfile.ZipFile("", "r") as zfile:
                zfile.extractall(firmware_dir.name)
        return firmware_dir


def execute(stages, args, ctx: InstallationContext):
    for stage in stages:
        if args.non_interactive:
            stage.read_context_from_params(args, ctx)
        else:
            is_continue = stage.ask_user_for_context(ctx)
            if not is_continue and ctx.abort:
                _logger.log(INFO, "Installation aborted.")
                return
        is_continue = stage.process(ctx)
        if not is_continue or ctx.abort:
            _logger.log(INFO, "Installation aborted.")
            return
    _logger.log(DEBUG, "Installation finished successfully!")
    print(f"{colorama.Fore.GREEN}Installation finished successfully!"
          f"{colorama.Style.RESET_ALL}")


def main():
    parser = argparse.ArgumentParser(description=
                                     "Installs ARRUS package.")
    parser.add_argument("--non_interactive", dest="non_interactive",
                        help="Whether to run this installation script in "
                             "non-interactive mode.",
                        type=bool, required=False, default=False)
    args = parser.parse_args()

    colorama.init()

    stages = [
        WelcomeStage(),
        FindExistingInstallationStage(),
        UnzipFilesStage(),
        UpdateEnvVariablesStage(),
        UpdateFirmwareStage()
    ]
    ctx = InstallationContext()
    return_code = 0
    try:
        execute(stages, args, ctx)
    except Exception as e:
        _logger.log(ERROR, "An exception occurred.")
        _logger.exception(e)
        _logger.log(INFO, "Installation aborted.")
        return_code = 1

    input("Press any key to exit.")
    ctx.cleanup()
    exit(return_code)


if __name__ == "__main__":
    main()

