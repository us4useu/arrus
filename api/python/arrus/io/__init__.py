"""
Common ARRUS I/O functions.
"""
from arrus.metadata import ConstMetadata


def read_metadata(filepath: str, format: str = None) -> ConstMetadata:
    """
    Reads the metadata from the file pointed by the filepath.

    NOTE: currently, only the pickle file format is supported. By default, 'pickle' file format will be used.

    NOTE: this method DO NOT the version field (we do not recommend saving
    the result of this function a new pickle file).

    :param filepath: path to the metadata file
    :param format: file format, default: pickle
    :return: arrus const metadata
    """

    if format is None:
        format = "pickle"

    if format == "pickle":
        # We are lazily importing the io.pickle, just to avoid
        # forcing pickle-related dependencies for the whole arrus
        # package.
        from arrus.io.pickle import read_pickled_metadata
        return read_pickled_metadata(filepath)
    else:
        raise ValueError(f"Unsupported file format: {format}")

