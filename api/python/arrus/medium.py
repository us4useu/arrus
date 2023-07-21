import dataclasses

@dataclasses.dataclass(frozen=True)
class Medium:
    name: str
    speed_of_sound: float
    attenuation_alpha: float
    attenuation_n: float

@dataclasses.dataclass(frozen=True)
class MediumDTO:
    name: str
    speed_of_sound: float

media = dict(
    (m.name, m) for m in [
        Medium(
            name="water",
            speed_of_sound=1490,
            attenuation_alpha=22e-4,
            attenuation_n=2,
        ),
        Medium(
            name="steel",
            speed_of_sound=5960,
            attenuation_alpha=None,
            attenuation_n=None,
        ),
        Medium(
            name="ats549",
            speed_of_sound=1450,
            attenuation_alpha=0.5,
            attenuation_n=1,
        ),
        Medium(
            name="soft_tissue",
            speed_of_sound=1540,
            attenuation_alpha=0.5,
            attenuation_n=1,
        ),
    ]
)

def get_medium_by_name(name: str):
    return media[name]

def get_media_list():
    return list(media.keys())

def list_media():
    print(get_media_list())

