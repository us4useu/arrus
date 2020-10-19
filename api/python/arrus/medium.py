import dataclasses


@dataclasses.dataclass(frozen=True)
class MediumDTO:
    name: str
    speed_of_sound: float