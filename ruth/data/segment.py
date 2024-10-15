import dataclasses
from typing import List, NewType, Tuple, Optional

SpeedKph = NewType("SpeedKph", float)
SpeedMps = NewType("SpeedMps", float)
LengthMeters = NewType("LengthMeters", float)
TravelTime = NewType("TravelTime", float)

Route = List[int]
RouteWithTime = Tuple[Route, Optional[TravelTime]]


@dataclasses.dataclass(frozen=True)
class SegmentPosition:
    # Index of the segment
    index: int
    # Position within the segment (in meters)
    position: LengthMeters


SegmentId = NewType("SegmentId", Tuple[int, int])


@dataclasses.dataclass(frozen=True)
class Segment:
    node_from: int
    node_to: int
    # Length of th segment (in meters)
    length: LengthMeters
    max_allowed_speed_kph: SpeedKph

    @property
    def id(self) -> SegmentId:
        return SegmentId((self.node_from, self.node_to))


def speed_mps_to_kph(speed_mps: SpeedMps) -> SpeedKph:
    return SpeedKph(speed_mps * 3.6)


def speed_kph_to_mps(speed_kph: SpeedKph) -> SpeedMps:
    return SpeedMps(speed_kph * (1000 / 3600))
