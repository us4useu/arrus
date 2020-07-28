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
import time
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
_FIRMWARE_SEA_FILE = f"us4OEM_tx_rev_{TX_FIRMWARE_VERSION[0].upper()}.sea"
_FIRMWARE_SED_FILE = f"us4OEM_tx_rev_{TX_FIRMWARE_VERSION[0].upper()}.sed"

_US4OEM_STATUS_BIN = "bin/us4OEMStatus.exe"
_US4OEM_FIRMWARE_UPDATE_EXE = "us4OEMFirmwareUpdate.exe"
_US4OEM_FIRMWARE_UPDATE_BIN = f"bin/{_US4OEM_FIRMWARE_UPDATE_EXE}"

_STATUS_US4OEM_KEY = "us4oems"


class InstallationContext:
    # tempfile.TemporaryDirectory()
    workspace_dir: object = None
    existing_install_dir: str = None
    install_dir: str = None
    abort: bool = False
    override_existing: bool = False
    system_status: dict = None
    arrus_bin_env_variables: dict = None
    n_us4oems: int = None

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

    def ask_options(self, msg: str, options: list, ctx: InstallationContext) -> str:
        options_str = [f"({i}) {option}" for i, option in enumerate(options)]
        options_str = ", ".join(options_str)
        while True:
            answer = input(f"{msg} [choose number: {options_str}]: ").strip()
            if not answer.isdigit():
                print(f"Please provide one of the numbers: {options_str}.")
            else:
                answer = int(answer)
                if answer < 0 or answer >= len(options):
                    print(f"Please provide one of the numbers: {options_str}.")
                else:
                    return options[answer]

    def ask_integer(self, msg: str, max: int, ctx: InstallationContext) -> int:
        while True:
            answer = input(f"{msg}: ").strip()
            if not answer.isdigit():
                print(f"Please provide positive number.")
            else:
                answer = int(answer)
                if answer < 1 or answer > max:
                    print(f"Provided number should be positive "
                          f"and lesser than: {max}.")
                else:
                    return answer

    def run_subprocess(self, args, context: InstallationContext):
        _logger.log(DEBUG, f"Running: {args} "
                           f"with env: {context.arrus_bin_env_variables}")
        process = subprocess.Popen(args, env=context.arrus_bin_env_variables)
        process.wait()
        return_code = process.returncode
        # TODO(pjarosik) consider capturing stderr info and log it into debug
        if return_code != 0:
            raise RuntimeError(f"The process {args} exited with code "
                               f"{return_code}")


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
        print(f"Starting {colorama.Fore.GREEN}ARRUS {PROJECT_VERSION}{colorama.Style.RESET_ALL} installer...")
        _logger.log(DEBUG, f"ARRUS: {PROJECT_VERSION},"
                           f"firmware version: {FIRMWARE_VERSION},"
                           f"tx firmware version: {TX_FIRMWARE_VERSION}")

        context.workspace_dir = tempfile.TemporaryDirectory()
        _logger.log(DEBUG, f"Workspace directory: {context.workspace_dir}")
        return True


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
                try:
                    full_path = os.path.join(path, expected_lib)
                    lib_file = Path(full_path)
                    if lib_file.is_file():
                        root_dir, lib_dir = os.path.split(path)
                        if lib_dir == _EXPECTED_LIB_DIR:
                            result.append(root_dir)
                except BaseException as e:
                    _logger.log(DEBUG, f"Error while checking path dirs: "
                                       f"{str(e)}")
        return result


class UnzipFilesStage(Stage):

    def read_context_from_params(self, args, ctx: InstallationContext):
        return True

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
            default_path = os.path.join(str(Path.home()), "arrus")
            path = self.ask_for_path(
                "Please provide destination path",
                default=default_path,
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
        _logger.log(INFO, f"Package installed in {ctx.install_dir}")
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
        return True

    def ask_user_for_context(self, ctx: InstallationContext):
        return True

    def process(self, ctx: InstallationContext):
        reg = winreg.ConnectRegistry(None, winreg.HKEY_CURRENT_USER)
        key = winreg.OpenKey(reg,r"Environment")
        current_user_path, d_type = winreg.QueryValueEx(key, "Path")

        path_to_add = os.path.join(
            ctx.install_dir,
            _EXPECTED_LIB_DIR
        )

        system_paths = current_user_path.split(os.pathsep)
        UpdateEnvVariablesStage.update_paths_list(system_paths, path_to_add)
        complete_value = os.pathsep.join(system_paths)
        _logger.log(DEBUG, f"New path to set in the system: {complete_value}")
        if len(complete_value) <= 1024:
            subprocess.check_output(["setx", "Path", complete_value])
        else:
            msg = f"{colorama.Fore.LIGHTRED_EX}" \
                  f"WARNING: Your current 'Path' environment variable is " \
                  f"longer than 1024 characters. Please add {path_to_add} " \
                  f"to your 'Path' environment variable manually." \
                  f"{colorama.Style.RESET_ALL}"
            print(msg)
            _logger.log(DEBUG, msg)

        # Update and set current path environment variable - for further
        # processing
        new_env = os.environ.copy()
        new_env["PATH"] = path_to_add + os.pathsep + new_env["PATH"]
        ctx.arrus_bin_env_variables = new_env
        _logger.log(INFO, "Updated environment variable 'Path'")
        return True

    @staticmethod
    def update_paths_list(paths_list, path_to_add):
        lib_path = Path(path_to_add)
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
        paths_list.insert(index, str(lib_path.resolve()))


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
        # Confirm to install software
        ans = self.ask_yn(f"Firmware update may take "
                          f"several hours. Are you sure you want to continue?",
                          ctx=ctx)
        if not ans:
            _logger.log(INFO, "To update hardware firmware re-run this "
                              "installator in the future (choose current "
                              "installation path).")
            return False
        # Ask for device type.
        systems = ["us4oem", "us4rlite"]

        system_type = self.ask_options("Choose device type",
                                       options=systems,
                                       ctx=ctx)
        if system_type == systems[0]:
            _logger.log(INFO, "Updating us4oems.")
            ctx.n_us4oems = self.ask_integer("Provide the number of "
                                                "Us4OEMs to update",
                                                max=2,
                                                ctx=ctx)
        elif system_type == systems[1]:
            _logger.log(INFO, "Updating us4rlite.")
            ctx.n_us4oems = 2
        else:
            raise ValueError(f"Unrecognized system type: {system_type}")
        return True

    def process(self, context: InstallationContext) -> bool:
        self.run_firmware_update_all(context)
        _logger.log(INFO, "Firmware updated. Restart the device and your "
                          "computer before using the system.")
        return True

    def run_firmware_update_all(self, context):
        update_bin = os.path.join(context.install_dir,
                                            _US4OEM_FIRMWARE_UPDATE_BIN)

        firmware_dir = self.unzip_firmware_if_necessary(context)
        firmware_dir = str(firmware_dir)
        # TODO(pjarosik) check why it is required for the update_bin
        # and .sea and .sed files in the same directory (otherwise the
        # TX firmware update does not work)
        shutil.copy2(update_bin, str(firmware_dir))
        old_wd = os.getcwd()
        try:
            os.chdir(firmware_dir)
            update_bin = _US4OEM_FIRMWARE_UPDATE_EXE

            n_us4oems = context.n_us4oems

            rpd_file_path = _FIRMWARE_RPD_FILE
            sea_file_path = _FIRMWARE_SEA_FILE
            sed_file_path = _FIRMWARE_SED_FILE

            call = [update_bin, "--rpd-file", rpd_file_path,
                    "--sea-file", sea_file_path,
                    "--sed-file", sed_file_path,
                    "--us4OEM-count", str(n_us4oems)]
            _logger.log(DEBUG, f"Will run: {call}")
            self.run_subprocess([update_bin,
                                "--rpd-file", rpd_file_path,
                                "--sea-file", sea_file_path,
                                "--sed-file", sed_file_path,
                                "--us4OEM-count", str(n_us4oems)],
                                context)
            time.sleep(5)
        finally:
            os.chdir(old_wd)

    def unzip_firmware_if_necessary(self, context: InstallationContext):
        firmware_dir = Path(os.path.join(context.workspace_dir.name, "firmware"))
        if not firmware_dir.is_dir():
            firmware_dir.mkdir(parents=True)
            with zipfile.ZipFile(_FIRMWARE_ZIP, "r") as zfile:
                zfile.extractall(str(firmware_dir))
        return firmware_dir


def execute(stages, args, ctx: InstallationContext):
    for stage in stages:
        if args.non_interactive:
            stage.read_context_from_params(args, ctx)
        else:
            is_continue = stage.ask_user_for_context(ctx)
            if ctx.abort:
                _logger.log(INFO, "Installation aborted.")
                return
            if not is_continue:
                return
        is_continue = stage.process(ctx)
        if ctx.abort:
            _logger.log(INFO, "Installation aborted.")
            return
        if not is_continue:
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

