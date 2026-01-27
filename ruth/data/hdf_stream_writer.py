from datetime import datetime, timedelta
import os
from typing import List
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
        # check if the file exists raise an error if it does
        if h5py.is_hdf5(filename):
            raise FileExistsError(f"The file {filename} already exists. Use a different filename or remove the existing file.")

        self.file = h5py.File(filename, 'x')

        if 'fcd' not in self.file or not isinstance(self.file['fcd'], h5py.Dataset):
            self.dataset = self.file.create_dataset(
                'fcd',
                shape=(0,),
                maxshape=(None,),
                dtype=compound_dtype,
                chunks=True
            )
        else:
            self.dataset = self.file['fcd']
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
            self.file.attrs['departure_time'] = departure_time.isoformat()
        if 'round_freq_s' not in self.file.attrs:
            self.file.attrs['round_freq_s'] = round_freq.total_seconds()
        self.file.flush()

    def append_file(self, buffer: List):
        # Create a structured numpy array from the FCD records in the buffer
        # Note: All simulation datetimes are treated as UTC (timezone-naive but UTC-based)
        # Using (datetime - epoch) avoids timezone issues across machines
        data = np.array([
            (int((fcd.datetime - _EPOCH).total_seconds()),
             fcd.segment.node_from, fcd.segment.node_to, fcd.segment.length,
             fcd.vehicle_id, float(fcd.offset_from_start), fcd.vehicle_speed_mps, fcd.active)
            for fcd in buffer
        ], dtype=compound_dtype)

        # Append data to the HDF5 file
        data_len = len(data)
        self.file["fcd"].resize((self.index + data_len,))
        self.file["fcd"][self.index:self.index + data_len] = data
        self.index += data_len
        self.file.flush()
        return data_len

    def close(self):
        self.file.close()
