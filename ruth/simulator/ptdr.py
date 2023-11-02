import os
from dataclasses import dataclass
from datetime import timedelta, datetime
from msgpack import Unpacker
from tqdm import tqdm
from typing import List


@dataclass
class LosAtTimeOfWeek:
    values: List[float]
    cumprobs: List[float]


@dataclass
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
        self.time_window_start = monday
        self.simulation_start_time = simulation_start_time
        self.period_size: timedelta = timedelta(days=7)
        self.time_slot_size: timedelta = timedelta(minutes=15)

    def get_timeslot_index(self, time_offset: timedelta):
        seconds_from_start = (self.simulation_start_time + time_offset - self.time_window_start).total_seconds()
        seconds_from_period_start = seconds_from_start % self.period_size.total_seconds()
        slot_time_id_in_period = int(seconds_from_period_start // self.time_slot_size.total_seconds())
        return slot_time_id_in_period


def load_ptdr(file_path: str) -> List[SegmentPTDRData]:
    if file_path is None:
        raise ValueError("PTDR file path is required for distributed route selection.")

    ptdr_data = []
    with open(file_path, "rb") as file:
        unpacker = tqdm(Unpacker(file), desc=os.path.basename(file_path), unit=" segments")
        for segment in unpacker:
            ptdr_data.append(SegmentPTDRData(
                id=segment['id'],
                length=segment['length'],
                max_speed=segment['max_speed'],
                profiles=[LosAtTimeOfWeek(values=profile['values'], cumprobs=profile['cumprobs']) for profile in
                          segment['profiles']]
            ))

    return ptdr_data
