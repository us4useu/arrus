"""
Utility functions to read python-pickled data.
"""
import dataclasses
from dataclasses import is_dataclass, fields
from arrus.metadata import ConstMetadata
from arrus.utils.common import is_package_available
from arrus.ops.us4r import (
    TxRxSequence
)
from arrus.devices.us4r import Us4RDTO

if not is_package_available("dill"):
    raise ValueError("Package 'dill' is required to read ARRUS "
                     "metadata from the Python-pickled files."
                     "Please install it first e.g. `pip install dill`")
import dill
import pickle
from typing import Dict, Type, Any, Iterable, Tuple


def read_pickled_metadata(filepath: str, force_version: str = None) -> ConstMetadata:
    """
    Reads pickled metadata from the filed pointed by the given filepath.

    NOTE: we do not recommend saving the result of this function to another file,
    otherwise you may not be able to read the newly-pickled
    file using this function (we are not modifying the version field here!)

    :param filepath: path to the pickle file
    :return: arrus const metadata
    """
    metadata = None
    # Read ARRUS version from the metadata file.
    with open(filepath, "rb") as f:
        metadata = dill.load(f)

    if force_version is not None:
        version = force_version
    else:
        version = metadata.version

    if version is None:
        raise ValueError(
            "Couldn't read metadata version from "
            f"'{filepath}' (unsupported legacy file?)."
        )
    # Get (major, minor) Ignore dev suffix
    major, minor = version.split(".")[:2]
    major, minor = int(major), int(minor)

    # Compare it with the current (major, minor)
    import arrus
    current_major, current_minor = arrus.__version__.split(".")[:2]
    current_major, current_minor = int(current_major), int(current_minor)

    # Strategy #0
    if (major, minor) == (current_major, current_minor):
        # just return the metadata
        return metadata

    # Do not support forward compatibility
    if (major, minor) > (current_major, current_minor):
        raise ValueError("ARRUS I/O currently do not provide "
                         "forward compatibility, however "
                         "you can try read the pickled file "
                         "using the `pickle` package directly.")

    # We have some older version of file -- use one of
    # import strategy.
    if (major, minor) not in _IMPORT_STRATEGIES:
        raise ValueError(f"The given metadata file (version: {version}) "
                         f"is no longer supported "
                         f"current ARRUS version: {arrus.__version__}.")
    strategy = _IMPORT_STRATEGIES.get((major, minor))
    if strategy is None:
        # No conversion is needed.
        return metadata
    # Read the file one more time, this time using the custom unpickler
    with open(filepath, "rb") as f:
        metadata = ArrusBackwardCompatibilityUnpickler(
            f, class_map=strategy.get_unpickler_class_map()
        ).load()
    # Apply strategies on the metadata fields (e.g. replace stubs with
    # the current version objects, wrap the field into a list, etc.).

    # Currently, we update only the context and data description
    # no other modifications are expected.
    context = _update_dataclass(
        metadata.context, strategy=strategy,
        current_name="context"
    )
    data_description = _update_dataclass(
        metadata.data_description, strategy=strategy,
        current_name="data_description"
    )
    return metadata.copy(context=context, data_desc=data_description)


## Unpickler
class ArrusBackwardCompatibilityUnpickler(pickle.Unpickler):
    def __init__(self, file, class_map: Dict[str, Type]):
        super().__init__(file)
        self.class_map = class_map

    def find_class(self, module, name):
        c = self.class_map.get(f"{module}.{name}")
        if c is not None:
            return c
        else:
            return super().find_class(module, name)


## MODEL
class FieldImportStrategy:

    def __init__(self):
        pass

    def accept(self, value: Any) -> bool:
        """
        Returns true if this strategy should be applied
        to the given field name and value.

        :param value: the value currently assigned to that field
        :return: True if the given field should be modified, false otherwise
        """
        return True

    def process(self, value: Any) -> Any:
        return value

    def get_unpickler_class(self) -> Tuple[str, Type]:
        """
        Returns a pair (source class absolute name -> dummy class),
        in case some dummy class should  be used for the given field,
        during the unpickling.

        Otherwise, returns None.
        """
        return None


class VersionImportStrategy:
    def __init__(self, field_strategies: Dict[str, FieldImportStrategy]):
        self.field_strategies = field_strategies

    def accept(self, field_name: str, value: Any) -> bool:
        strategy = self.field_strategies.get(field_name)
        return strategy is not None and strategy.accept(value)

    def get_unpickler_class_map(self) -> Dict[str, Type]:
        """
        Returns full class name -> target Class object map.

        The returned value can be used in the ARRUS unpickler
        to replace the old classes with the stubs.
        """
        result = {}
        for strategy in self.field_strategies.values():
            r = strategy.get_unpickler_class()
            if r is not None:
                abs_name, cls = r
                result[abs_name] = cls
        return result

    def get_field_strategy(self, field_name: str) -> FieldImportStrategy:
        """
        Returns field strategy for the given field name.
        """
        return self.field_strategies.get(field_name)


# ARRUS 0.10.0 and 0.9.0.
class Arrus010SchemeTxRxSequenceImportStrategy(FieldImportStrategy):

    def __init__(self):
        super().__init__()

    def accept(self, value: Any) -> bool:
        # We need to process only the non-iterable values (e.g. raw TxRxSequence)
        return isinstance(value, TxRxSequence)

    def process(self, value: Any) -> Any:
        """
        Conerts to a single-element tuple (provided the support for multi-sequence schemes.
        :param value:
        :return:
        """
        return (value, )


@dataclasses.dataclass
class Arrus010Us4RDTO:
    probe: object
    sampling_frequency: float


class Arrus010Us4RDTOImportStrategy(FieldImportStrategy):
    def __init__(self):
        super().__init__()

    def accept(self, value: Any) -> bool:
        return True

    def process(self, value: Arrus010Us4RDTO) -> Us4RDTO:
        """
        Converts to a single-element tuple (provided the support for multi-sequence schemes.
        """
        return Us4RDTO(
            probe=value.probe,
            sampling_frequency=value.sampling_frequency,
            # Only us4OEM 65 MHz systems could be used with ARRUS 0.10.0 or earlier
            data_sampling_frequency=65e6,
        )

    def get_unpickler_class(self) -> Tuple[str, Type]:
        return "arrus.devices.us4r.Us4RDTO", Arrus010Us4RDTO


class Arrus010ImportStrategy(VersionImportStrategy):
    def __init__(self):
        super().__init__(
            {
                "context.device": Arrus010Us4RDTOImportStrategy()
                # No longer needed.
                # "context.sequence": Arrus010SchemeTxRxSequenceImportStrategy(),
                # "context.raw_sequence": Arrus010SchemeTxRxSequenceImportStrategy()
            }
        )


_IMPORT_STRATEGIES = {
    # metadata (major, minor)
    (0, 10): Arrus010ImportStrategy(),
    # No non-backward compatible changes between 0.11.0 and 0.12.0
    (0, 11): None
}


def _update_dataclass(obj: Any, strategy: VersionImportStrategy, current_name=None) -> Any:
    if not is_dataclass(obj):
        return obj

    kwargs = {}
    for f in fields(obj):
        value = getattr(obj, f.name)
        abs_field_name = f.name if not current_name else f"{current_name}.{f.name}"
        if strategy.accept(abs_field_name, value):
            field_strategy = strategy.get_field_strategy(abs_field_name)
            kwargs[f.name] = field_strategy.process(value)
        else:
            if is_dataclass(value):
                kwargs[f.name] = _update_dataclass(
                    value, strategy, current_name=abs_field_name
                )
            elif isinstance(value, list):
                kwargs[f.name] = [
                    _update_dataclass(v, strategy=strategy, current_name=current_name)
                    if is_dataclass(v) else v for v in value
                ]
            elif isinstance(value, dict):
                kwargs[f.name] = {
                    k: _update_dataclass(v, strategy=strategy, current_name=current_name)
                    if is_dataclass(v) else v
                    for k, v in value.items()
                }
            else:
                kwargs[f.name] = value

    return dataclasses.replace(obj, **kwargs)
