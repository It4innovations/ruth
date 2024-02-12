from collections import defaultdict
from typing import List, TYPE_CHECKING, Dict

import pandas as pd

from .data.segment import SegmentId

if TYPE_CHECKING:
    from .simulator.simulation import FCDRecord


class FCDHistory:
    """
    Stores historical FCD records.
    """

    def __init__(self):
        self.fcd_history: List["FCDRecord"] = []
        self.fcd_by_segment: Dict[SegmentId, List[FCDRecord]] = defaultdict(list)

    def add(self, fcd: "FCDRecord"):
        self.fcd_by_segment[fcd.segment.id].append(fcd)
        self.fcd_history.append(fcd)

    def to_dataframe(self):  # todo: maybe process in chunks
        data = defaultdict(list)
        for fcd in self.fcd_history:
            data["timestamp"].append(fcd.datetime)
            data["node_from"].append(fcd.segment.node_from)
            data["node_to"].append(fcd.segment.node_to)
            data["segment_length"].append(fcd.segment.length)
            data["vehicle_id"].append(fcd.vehicle_id)
            data["start_offset_m"].append(fcd.start_offset)
            data["speed_mps"].append(fcd.speed)
            data["status"].append(fcd.status)
            data["active"].append(fcd.active)

        return pd.DataFrame(data)

    def speed_in_time_at_segment(self, datetime, node_from, node_to):
        speeds = [fcd.speed for fcd in self.fcd_by_segment.get(SegmentId((node_from, node_to)), []) if
                  fcd.datetime == datetime]
        if len(speeds) == 0:
            return None
        return sum(speeds) / len(speeds)

    def __getstate__(self):
        self.fcd_history.sort(key=lambda fcd: (fcd.datetime, fcd.segment.id))
        return self.fcd_history

    def __setstate__(self, state):
        self.fcd_history = state
