from datetime import datetime, timedelta
import os
from typing import List
import os
import tempfile
import h5py
import numpy as np

from .map import Map

# Unix epoch for timezone-independent timestamp calculation
_EPOCH = datetime(1970, 1, 1)

# Define the compound dtype for HDF5
compound_dtype = np.dtype([
    ("timestamp", np.int64),  # Timestamp in seconds (UTC)
    ("node_from", np.int64),
    ("node_to", np.int64),
    ("segment_length", np.int32),
    ("vehicle_id", np.int64),
    ("start_offset_m", np.float32),
    ("speed_mps", np.float32),
    ("active", np.bool_),
])

class HDF5Writer:
    def __init__(self, filename, dtype=None):
        # If the path exists but is not a valid HDF5 file, raise to avoid data corruption
        if os.path.exists(filename):
            raise FileExistsError(f"The path {filename} exists.")

        # Open file in append mode so we can continue writing to an existing file
        self.file = h5py.File(filename, 'a')

        # Create or get the dataset; require_dataset will validate dtype if it exists
        # Use a reasonable chunk size for growing 1D records
        chunk_shape = (1024,)
        self.dataset = self.file.require_dataset('fcd', shape=(0,), maxshape=(None,), dtype=compound_dtype,
                                                chunks=chunk_shape)
        self.index = self.dataset.shape[0]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def save_computational_time(self, computational_time: float):
        if 'computational_time' not in self.file.attrs:
            self.file.attrs['computational_time'] = computational_time
            self.file.flush()

    def save_map(self, routing_map: Map, departure_time: datetime, round_freq: timedelta):
        if 'bbox' not in self.file.attrs:
            self.file.attrs['bbox'] = tuple(routing_map.bbox.get_coords())
        if 'download_date' not in self.file.attrs:
            self.file.attrs['download_date'] = str(routing_map.download_date)
        if 'departure_time' not in self.file.attrs:
            # store as ISO string
            self.file.attrs['departure_time'] = departure_time.isoformat()
        if 'round_freq_s' not in self.file.attrs:
            self.file.attrs['round_freq_s'] = float(round_freq.total_seconds())
        self.file.flush()

    def append_file(self, buffer: List):
        data = np.array([
            (
                int((fcd.datetime - _EPOCH).total_seconds()),
                int(fcd.segment.node_from),
                int(fcd.segment.node_to),
                int(fcd.segment.length),
                int(fcd.vehicle_id),
                float(fcd.offset_from_start),
                float(fcd.vehicle_speed_mps),
                bool(fcd.active)
            )
            for fcd in buffer
        ], dtype=compound_dtype)

        data_len = len(data)
        if data_len == 0:
            return 0

        # Resize dataset and append
        self.file['fcd'].resize((self.index + data_len,))
        self.file['fcd'][self.index:self.index + data_len] = data
        self.index += data_len
        self.file.flush()
        return data_len

    def close(self):
        # Close the HDF5 file
        try:
            self.file.close()
        except Exception:
            pass
