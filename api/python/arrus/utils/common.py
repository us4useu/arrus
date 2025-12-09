import importlib

def is_package_available(package_name):
    return importlib.util.find_spec(package_name) is not None