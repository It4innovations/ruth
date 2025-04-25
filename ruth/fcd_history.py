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

    def to_dataframe_short(self):
        data = {
            "timestamp": [fcd.datetime for fcd in self.fcd_history],
            "node_from": pd.array([fcd.segment.node_from for fcd in self.fcd_history], dtype="Int64"),
            "node_to": pd.array([fcd.segment.node_to for fcd in self.fcd_history], dtype="Int64"),
            "segment_length": pd.array([fcd.segment.length for fcd in self.fcd_history], dtype="float"),
            "vehicle_id": pd.array([fcd.vehicle_id for fcd in self.fcd_history], dtype="Int32"),
            "start_offset_m": pd.array([fcd.start_offset for fcd in self.fcd_history], dtype="float"),
            "speed_mps": pd.array([fcd.speed for fcd in self.fcd_history], dtype="float"),
        }

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
