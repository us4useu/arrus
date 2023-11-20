from typing import List
import re

import numpy as np


class Constant:
    """
    ARRUS constant. The value represents some ndarray to be stored on the device
    pointed by the placement id.
    """

    def __init__(self, value, placement, name):
        self.value = value
        self.placement = placement
        self.name = name

    def get_full_name(self):
        return self.placement + "/" + self.name


def _get_unique_name(constants: List[Constant], name):
    """
    NOTE: this function is deprecated and needed only for new ARRUS framework
    API transition. Only for internal use!
    """
    name_pattern = fr"{name}:([0-9]+)"
    numbers = []
    for constant in constants:
        if constant.placement != "/Us4R:0":
            raise ValueError("Currently the constants can be defined only for "
                             "/Us4R:0 device.")
        match = re.search(name_pattern, constant.name)
        if match:
            id = match.group(1)
            numbers.append(int(id))
        else:
            raise ValueError(f"Invalid name, expected: {name}:number")

    new_number = np.max(numbers) + 1
    return f"{name}:{new_number}"
