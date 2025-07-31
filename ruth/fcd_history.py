import logging
from collections import defaultdict, deque
from datetime import datetime
from typing import List, TYPE_CHECKING, Dict
import pandas as pd
from .data.hdf_stream_writer import HDF5Writer
from .data.segment import SegmentId
from .mpi_comm.distributor import MPIDistributor

if TYPE_CHECKING:
    from .simulator.simulation import FCDRecord


class FCDHistory:

    def __init__(self, h5_path: str, buffer_size, keep_in_memory):
        if not MPIDistributor.is_master(): return
        self.path = h5_path
        self.buffer_size = buffer_size
        self.buffer: List[FCDRecord] = []

        self.keep_in_memory = keep_in_memory
        self.fcd_history: List[FCDRecord] = []
        self.start_time = None
        self.writer = None

    def __enter__(self):
        if self.writer is None:
            self.writer = HDF5Writer(self.path)
        self.start_time = datetime.now()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not MPIDistributor.is_master(): return
        if self.start_time:
            computational_time = (datetime.now() - self.start_time).total_seconds()
            self.writer.save_computational_time(computational_time)
        self.close()

    # if pickling do not pickle the writer
    def __getstate__(self):
        state = self.__dict__.copy()
        if 'writer' in state:
            del state['writer']
        return state

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
            self.writer = None
        elif isinstance(state, list):
            # backward compatibility for old pickles
            self.keep_in_memory = True
            self.fcd_history = state
        else:
            print(state)
            raise TypeError(f"Expected dict for state, got {type(state)}")


    def extend(self, fcd: List["FCDRecord"]):
        if not MPIDistributor.is_master(): return
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
        logging.warning("This function will be deprecated soon with migration to h5 storage for FCD history.")
        if not self.keep_in_memory:
            raise NotImplementedError("to_dataframe is disabled when streaming to HDF5.")

        data = defaultdict(list)
        for fcd in self.fcd_history:
            data["timestamp"].append(fcd.datetime)
            data["node_from"].append(fcd.segment.node_from)
            data["node_to"].append(fcd.segment.node_to)
            data["segment_length"].append(fcd.segment.length)
            data["vehicle_id"].append(fcd.vehicle_id)
            data["start_offset_m"].append(fcd.offset_from_start)
            data["speed_mps"].append(fcd.vehicle_speed_mps)
            data["status"].append(fcd.status)
            data["active"].append(fcd.active)

        return pd.DataFrame(data)