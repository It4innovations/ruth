import dataclasses
from typing import List, NewType

SpeedKph = NewType("SpeedKph", float)
SpeedMps = NewType("SpeedMps", float)
LengthMeters = NewType("LengthMeters", float)

Route = List[int]


@dataclasses.dataclass(frozen=True)
class SegmentPosition:
    # Index of the segment
    index: int
    # Position within the segment (in meters)
    position: LengthMeters


@dataclasses.dataclass(frozen=True)
class Segment:
    id: str
    # Length of th segment (in meters)
    length: LengthMeters
    max_allowed_speed_kph: SpeedKph


def speed_mps_to_kph(speed_mps: SpeedMps) -> SpeedKph:
    return SpeedKph(speed_mps * 3.6)


def speed_kph_to_mps(speed_kph: SpeedKph) -> SpeedMps:
    return SpeedMps(speed_kph * (1000 / 3600))
