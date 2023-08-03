import dataclasses

@dataclasses.dataclass(frozen=True)
class Medium:
    """
    Medium description class.
    The attenuation parameters are associated with the assumption that
    atteuation coefficient alpha is given by following equation

                    alpha = a*(f)^n

    where a is attenuation coefficient at 1MHz, f is frequency (in MHz),
    and n is dimensionless exponent.
    NOTE: In arrus SI units are used.
    :param name: medium unique name
    :param speed_of_sound: longitudinal wave propagation speed in [m/s]
    :param attenuation_a: attenuation coefficient at 1MHz in [dB/Hz/m]
    :param atteunation_n: dimensionless exponent determining the attenuation frequency dependence
    """
    name: str
    speed_of_sound: float
    attenuation_a: float
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
            attenuation_a=22e-4,
            attenuation_n=2,
        ),
        Medium(
            name="steel",
            speed_of_sound=5960,
            attenuation_a=None,
            attenuation_n=None,
        ),
        Medium(
            name="ats549",
            speed_of_sound=1450,
            attenuation_a=0.5e-4,
            attenuation_n=1,
        ),
        Medium(
            name="soft_tissue",
            speed_of_sound=1540,
            attenuation_a=0.5e-4,
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

