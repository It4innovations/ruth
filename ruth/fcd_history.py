from collections import defaultdict, deque
from typing import List, TYPE_CHECKING, Dict
import pandas as pd
from .data.hdf_stream_writer import HDF5Writer
from .data.segment import SegmentId

if TYPE_CHECKING:
    from .simulator.simulation import FCDRecord


class FCDHistory:

    def __init__(self, h5_path: str, buffer_size, keep_in_memory):
        self.path = h5_path
        self.writer = HDF5Writer(h5_path)
        self.buffer_size = buffer_size
        self.buffer: List[FCDRecord] = []

        self.keep_in_memory = keep_in_memory
        self.fcd_history: List[FCDRecord] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # if pickling do not pickle the writer
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['writer']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.writer = HDF5Writer(self.path)

    def extend(self, fcd: List["FCDRecord"]):
        self.buffer.extend(fcd)

        if self.keep_in_memory:
            self.fcd_history.extend(fcd)

        if len(self.buffer) >= self.buffer_size:
            self.flush_to_disk()

    def flush_to_disk(self):
        if not self.buffer: return
        self.writer.append_file(self.buffer)
        self.buffer.clear()

    def close(self):
        if self.buffer:
            self.flush_to_disk()
        self.writer.close()

    def to_dataframe(self):
        if not self.keep_in_memory:
            raise NotImplementedError("to_dataframe is disabled when streaming to HDF5.")

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
        if not self.keep_in_memory:
            raise NotImplementedError("speed_in_time_at_segment is disabled when streaming to HDF5.")

        # check if has attribute fcd_by_segment
        if not hasattr(self, 'fcd_by_segment'):
            self.fcd_by_segment = defaultdict(list)
            for fcd in self.fcd_history:
                self.fcd_by_segment[fcd.segment.id].append(fcd)

        speeds = [fcd.speed for fcd in self.fcd_by_segment.get(SegmentId((node_from, node_to)), []) if
                  fcd.datetime == datetime]
        if len(speeds) == 0:
            return None
        return sum(speeds) / len(speeds)