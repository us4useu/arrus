import sys
import types
import importlib
import os

def mock_import(module_name: str, **kwargs):
    mock_module = types.ModuleType(module_name)
    module_path = module_name.split(".")
    if len(module_path) > 1:
        parent = importlib.import_module(".".join(module_path[:-1]))
        setattr(parent, module_path[-1], mock_module)
    sys.modules[module_name] = mock_module
    for key, value in kwargs.items():
        setattr(mock_module, key, value)
    return mock_module

def get_dataset_path(path):
    """
    Returns an absolute path to a given arrus dataset file.

    "ARRUS_DATASET_PATH' environment variable should be declared and set
    to the path, where arrus dataset is located.

    :param path: path to the file to load
    :return: path to a given arrus dataset file
    """
    key = "ARRUS_DATASET_PATH"
    dataset_path = os.environ.get(key, None)
    if dataset_path is None:
        raise KeyError("Environment variable %s should be declared." % key)
    return os.path.join(dataset_path, path)


