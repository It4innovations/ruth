from collections import defaultdict, deque
from typing import List, TYPE_CHECKING, Dict
import numpy as np
import pandas as pd
from ruth.data.hdf_stream_writer import HDF5Writer
from .data.segment import SegmentId

if TYPE_CHECKING:
    from .simulator.simulation import FCDRecord


class FCDHistory:

    def fcd_to_array(self, fcd):
        return {
            "timestamp": fcd.datetime.isoformat(),
            "vehicle_id": fcd.vehicle_id,
            "segment_id": int(hash(fcd.segment.id)),  # changed
            "start_offset": fcd.start_offset,
            "speed": fcd.speed,
            "status": fcd.status,
            "active": int(fcd.active)
        }

    def __init__(self, h5_path: str, buffer_size=0, data_shape=(9,)):
        self.writer = HDF5Writer(h5_path)
        self.buffer_size = buffer_size
        self.fcd_history = deque(maxlen=buffer_size) if buffer_size > 0 else None
        self.fcd_by_segment: Dict[SegmentId, List['FCDRecord']] = defaultdict(list)

    








    def add(self, fcd: "FCDRecord"):
        self.fcd_by_segment[fcd.segment.id].append(fcd)

        record = self.fcd_to_array(fcd)

        # # DEBUG PRINT — inspect field values and types
        # print("FCD RECORD DEBUG")
        # for k, v in record.items():
        #     print(f"  {k}: {v} ({type(v)})")

        self.writer.append(record)

        if self.fcd_history is not None:
            self.fcd_history.append(fcd)





    def to_dataframe(self):
        raise NotImplementedError("to_dataframe is disabled when streaming to HDF5.")

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
        return list(self.fcd_history) if self.fcd_history is not None else []

    def __setstate__(self, state):
        self.fcd_history = deque(state)
