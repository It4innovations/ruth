import dataclasses

SpeedKph = float
LengthMeters = float


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
