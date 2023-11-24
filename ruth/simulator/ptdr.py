from dataclasses import dataclass
from datetime import timedelta, datetime
from typing import List


@dataclass(frozen=True)
class LosAtTimeOfWeek:
    values: List[float]
    cumprobs: List[float]


@dataclass(frozen=True)
class SegmentPTDRData:
    id: str
    length: float
    max_speed: float
    profiles: List[LosAtTimeOfWeek]

    def __post_init__(self):
        assert len(self.profiles) == 672


class PTDRInfo:
    def __init__(self, simulation_start_time: datetime):
        monday = simulation_start_time - timedelta(days=simulation_start_time.weekday())
        monday = monday.replace(hour=0, minute=0, second=0)
        self.time_window_start = monday
        self.simulation_start_time = simulation_start_time
        self.period_size: timedelta = timedelta(days=7)

    def get_time_from_start_of_interval(self, time_offset: timedelta):
        """
        Returns the number of milliseconds for the given `time_offset` from the beginning of Monday.
        """
        seconds_from_start = (self.simulation_start_time + time_offset - self.time_window_start).total_seconds()
        seconds_from_period_start = seconds_from_start % self.period_size.total_seconds()
        return seconds_from_period_start * 1000
