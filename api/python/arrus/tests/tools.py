import sys
import types
import importlib

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

