import logging
from datetime import datetime
from typing import List, TYPE_CHECKING
import os
import glob

import numpy as np

from .data.hdf_stream_writer import HDF5Writer

if TYPE_CHECKING:
    from .simulator.simulation import FCDRecord


class FCDHistory:

    def __init__(self, h5_path: str, buffer_size, max_records_per_file=None):
        self.base_path = h5_path
        self.buffer_size = buffer_size
        self.buffer: List[FCDRecord] = []

        self.fcd_history: List[FCDRecord] = []
        self.start_time = None
        self.writer = None

        self.max_records_per_file = max_records_per_file
        if self.max_records_per_file is None:
            self.max_records_per_file = np.max
        self._current_part = 0

    def __enter__(self):
        if self.writer is None:
            self._open_existing_writer()
        self.start_time = datetime.now()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            computational_time = (datetime.now() - self.start_time).total_seconds()
            self.writer.save_computational_time(computational_time)
        self.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'writer' in state:
            del state['writer']
        return state

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
            self.writer = None
            if not hasattr(self, 'base_path'):
                self.base_path = self.path
            if not hasattr(self, "_current_part"):
                self._current_part = 0
        else:
            print(state)
            raise TypeError(f"Expected dict for state, got {type(state)}")


    def extend(self, fcd: List["FCDRecord"]):
        self.buffer.extend(fcd)

        if len(self.buffer) >= self.buffer_size:
            if self.writer is None:
                self._open_existing_writer()
            self.flush_to_disk()

    def flush_to_disk(self):
        if not self.buffer:
            return

        # while there is data in buffer, append as much as fits in current file; rotate if needed
        while self.buffer:
            remaining_in_file = self.max_records_per_file - self.writer.index
            if remaining_in_file <= 0:
                self._rotate_writer()
                continue

            to_take = min(len(self.buffer), remaining_in_file)
            chunk = self.buffer[:to_take]
            self.writer.append_file(chunk)
            # remove chunk from buffer
            del self.buffer[:to_take]

    def close(self):
        if self.buffer:
            self.flush_to_disk()
        if self.writer is not None:
            self.writer.close()

    def to_dataframe(self):
        logging.warning("This function will be deprecated soon with migration to h5 storage for FCD history.")
        raise NotImplementedError("to_dataframe is disabled when streaming to HDF5.")

    def _open_existing_writer(self):
        base_no_ext = os.path.splitext(self.base_path)[0]
        new_path = f"{base_no_ext}-part{self._current_part:04d}.h5"
        self.path = new_path
        self.writer = HDF5Writer(self.path)

    def _rotate_writer(self):
        if self.writer is not None:
            self.writer.close()

        self._current_part = self._current_part + 1

        base_no_ext = os.path.splitext(self.base_path)[0]
        new_path = f"{base_no_ext}-part{self._current_part:04d}.h5"
        self.path = new_path
        self.writer = HDF5Writer(self.path)
