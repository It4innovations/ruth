from dataclasses import dataclass
from typing import List


@dataclass
class LosAtTimeOfWeek:
    values: List[float]
    cumprobs: List[float]


@dataclass
class SegmentPTDRData:
    id: int
    length: float
    max_speed: float
    profiles: List[LosAtTimeOfWeek]

    def __post_init__(self):
        assert len(self.profiles) == 672
