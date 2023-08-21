import dataclasses
from typing import List

SpeedKph = float
SpeedMps = float
LengthMeters = float

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
