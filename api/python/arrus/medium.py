import dataclasses

@dataclasses.dataclass(frozen=True)
class Medium:
    name: str
    speed_of_sound: float

@dataclasses.dataclass(frozen=True)
class MediumDTO:
    name: str
    speed_of_sound: float

media = dict(
    (m.name, m) for m in [
        Medium(name="water", speed_of_sound=1490),
        Medium(name="steel", speed_of_sound=5960),
        Medium(name="ats549", speed_of_sound=1450),
    ]
)


def get_medium_by_name(name: str):
    return media[name]

