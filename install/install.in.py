import abc
import logging
from logging import INFO, DEBUG, WARNING, ERROR
import colorama
import os
import argparse
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


class InstallationContext:
    existing_install_dir: str = None
    install_dir: str = None
    abort: bool = False


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

    def ask_yn(self, msg: str, ctx: InstallationContext) -> bool:
        while True:
            answer = input(f"{msg} [Yes/No/Quit]: ").lower().strip()
            if answer in {"yes", "y"}:
                return True
            elif answer in {"no", "n"}:
                return False
            elif answer in {"quit", "q"}:
                ctx.abort = True
                return None
            else:
                print("Please answer Yes/No/Quit")

    def ask_for_path(self, msg: str, default, ctx: InstallationContext) -> bool:
        while True:
            answer = input(f"{msg} [default:{default}]: ").strip()
            if answer:

                # Check if user has permission to create this directory (and the path exists)
                # Check if the directory already exists and there is something
                # in it
                return answer



class WelcomeStage(Stage):

    def read_context_from_params(self, args, ctx: InstallationContext) -> bool:
        return True

    def ask_user_for_context(self, ctx: InstallationContext) -> bool:
        return True

    def process(self, context: InstallationContext) -> bool:
        _logger.log(INFO, f"Starting "
                    f"{colorama.Fore.YELLOW}ARRUS "
                    f"{PROJECT_VERSION}{colorama.Style.RESET_ALL}"
                    f"installer...")
        return True


class FindExistingInstallationStage(Stage):

    def read_context_from_params(self, args, ctx: InstallationContext) -> bool:
        return True

    def ask_user_for_context(self, ctx: InstallationContext) -> bool:
        return True

    def process(self, context: InstallationContext) -> bool:
        _logger.log(INFO, "Checking for existing installations...")
        existing_installations = self._find_arrus_paths()

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
    UNZIP_DEFAULT_DIR = "C:\local\arrus"

    def read_context_from_params(self, args, ctx: InstallationContext):
        pass

    def ask_user_for_context(self, ctx: InstallationContext):
        if ctx.existing_install_dir is not None:
            msg = f"Found ARRUS in path '{ctx.existing_install_dir}'. " \
                  f"Would you like to update it?"
            is_override = self.ask_yn(msg, ctx)
            if is_override is None and ctx.abort:
                return False
            if is_override:
                ctx.install_dir = ctx.existing_install_dir
                return True
        # ARRUS wasn't already installed or the user want to install in some
        # other place.
        path = self.ask_for_path(
            "Choose destination directory",
            default=UnzipFilesStage.UNZIP_DEFAULT_DIR,
            ctx=ctx
        )
        ctx.install_dir = path


    def process(self, ctx: InstallationContext):
        # Extract arrus.zip file to the provided directory.
        pass



class UpdateEnvVariablesStage(Stage):

    def process(self, context: InstallationContext):
        return True


def main():
    non_interactive = False

    parser = argparse.ArgumentParser(description="Installs ARRUS package in the system.")
    parser.add_argument("--non_interactive", dest="non_interactive",
                        help="Whether to run this installation script in "
                             "non-interactive mode.",
                        type=bool, required=False, default=False)
    args = parser.parse_args()

    colorama.init()
    stages = [
        WelcomeStage()
    ]

    ctx = InstallationContext()

    # TODO catch any errors that happened here

    for stage in stages:
        if args.non_interactive:
            stage.read_context_from_params(args, ctx)
        else:
            is_continue = stage.ask_user_for_context(ctx)
            if not is_continue and ctx.abort:
                _logger.log(INFO, "Installation aborted.")
        is_continue = stage.process(ctx)
        if not is_continue and ctx.abort:
            _logger.log(INFO, "Installation aborted.")
            exit(1)


if __name__ == "__main__":
    main()

