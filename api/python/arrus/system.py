import abc
import dataclasses


class SystemCfg(abc.ABC):
    """
    Description of the system with which the user wants to communicate.

    """
    pass

@dataclasses.dataclass(frozen=True)
class CustomUs4RCfg(SystemCfg):
    """
    Description of the custom Us4R system.

    :param n_us4oems: number of us4oem modules a user wants to use
    :param is_hv256: is the hv256 voltage supplier available?
    :param master_us4oem: index of the master Us4OEM module. A master module \
        triggers TxRx execution and should be connected with the voltage \
        supplier (if available). By default equal to 0.
    """
    n_us4oems: int
    is_hv256: bool = False
    master_us4oem: int = 0
