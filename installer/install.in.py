import abc
import logging
from logging import INFO, DEBUG, WARNING, ERROR
import colorama
import os
import argparse
import shutil
import zipfile
from pathlib import Path


PROJECT_VERSION = "${PROJECT_VERSION}"

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

_EXPECTED_LIB_DIR = "lib64"
_EXPECTED_LIBS = ["Us4OEM.dll"]
_PKG_ZIP = "arrus.zip"


class InstallationContext:
    existing_install_dir: str = None
    install_dir: str = None
    abort: bool = False
    override_existing: bool = False


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


class WelcomeStage(Stage):

    def read_context_from_params(self, args, ctx: InstallationContext) -> bool:
        return True

    def ask_user_for_context(self, ctx: InstallationContext) -> bool:
        return True

    def process(self, context: InstallationContext) -> bool:
        _logger.log(INFO, f"Starting "
                    f"{colorama.Fore.YELLOW}ARRUS "
                    f"{PROJECT_VERSION}{colorama.Style.RESET_ALL} "
                    f"installer...")
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
        if ctx.existing_install_dir is not None:
            msg = f"Found ARRUS in path '{ctx.existing_install_dir}'. " \
                  f"Would you like to replace it with newer version?\n" \
                  f"{colorama.Fore.LIGHTRED_EX}" \
                  f"WARNING: " \
                  f"The contents of the destination directory will be deleted!"\
                  f"{colorama.Style.RESET_ALL}"
            is_override = self.ask_ynq(msg, ctx)
            if is_override is None and ctx.abort:
                return False
            if is_override:
                ctx.install_dir = ctx.existing_install_dir
                ctx.override_existing = True
                return True
        # ARRUS wasn't already installed or the user want to install in some
        # other place.
        path = self.ask_for_path(
            "Please provide destination path",
            default=UnzipFilesStage.UNZIP_DEFAULT_DIR,
            ctx=ctx
        )
        if os.path.exists(path):
            if os.path.isfile(path):
                msg = f"WARNING: The FILE '{path}' will be deleted! " \
                      f"Are you sure you want to continue?"
            elif os.path.isdir(path):
                msg = f"WARNING: The contents of '{path}' will be deleted! " \
                      f"Are you sure you want to continue?"
            else:
                raise ValueError(f"Unrecognized file system object: {path}")
            msg = f"{colorama.Fore.LIGHTRED_EX}{msg}{colorama.Style.RESET_ALL}"
            result = self.ask_yn(msg, ctx)
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

    def process(self, context: InstallationContext):
        return True


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
        if not is_continue and ctx.abort:
            _logger.log(INFO, "Installation aborted.")
            return
    _logger.log(INFO,
                f"{colorama.Fore.GREEN}Installation finished successfully!"
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
        UnzipFilesStage()
    ]

    ctx = InstallationContext()
    try:
        execute(stages, args, ctx)
    except Exception as e:
        _logger.log(ERROR, "An exception occurred.")
        _logger.exception(e)
        _logger.log(INFO, "Installation aborted.")
        exit(1)


if __name__ == "__main__":
    main()

