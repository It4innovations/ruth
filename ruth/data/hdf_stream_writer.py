from typing import List
import h5py
import numpy as np

# Define the compound dtype for HDF5
compound_dtype = np.dtype([
    ("timestamp", np.int64),  # Timestamp in nanoseconds
    ("node_from", np.int64),
    ("node_to", np.int64),
    ("segment_length", np.int32),
    ("vehicle_id", np.int64),
    ("start_offset_m", np.float32),
    ("speed_mps", np.float32),
    ("active", np.bool_),
])

# Define validation rules for each field in the compound dtype
field_validations = {
    "timestamp": {"type": np.int64, "range": (-np.iinfo(np.int64).max, np.iinfo(np.int64).max)},
    "node_from": {"type": np.int64, "range": (-np.iinfo(np.int64).max, np.iinfo(np.int64).max)},
    "node_to": {"type": np.int64, "range": (-np.iinfo(np.int64).max, np.iinfo(np.int64).max)},
    "segment_length": {"type": np.int32, "range": (0, np.iinfo(np.int32).max)},
    "vehicle_id": { "type": np.int64, "range": (0, np.iinfo(np.int64).max)},
    "start_offset_m": {"type": np.float32, "range": (0, np.finfo(np.float32).max)},
    "speed_mps": {"type": np.float32, "range": (0, np.finfo(np.float32).max)},
    "active": {"type": np.bool_, "range": (False, True)},
}

def validate_data(data):
    # Validate each field in the structured numpy array
    for field in data.dtype.names:  # Iterate through the field names of the structured array
        field_info = field_validations.get(field)

        # Get the value from the structured array
        value = data[field]

        # Type validation
        if not isinstance(value, field_info["type"]):
            raise ValueError(f"Invalid type for {field}: Expected {field_info['type']}, got {type(value)}")

        # Range validation
        min_val, max_val = field_info["range"]
        if not (min_val <= value <= max_val):
            raise ValueError(f"Invalid value for {field}: {value} outside range ({min_val}, {max_val})")

    return True

class HDF5Writer:
    def __init__(self, filename, dtype=None):
        self.file = h5py.File(filename, 'a')

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

    def append_file(self, buffer: List):
        # Create a structured numpy array from the FCD records in the buffer
        data = np.array(
            [(int(fcd.datetime.timestamp()), fcd.segment.node_from, fcd.segment.node_to, fcd.segment.length,
              fcd.vehicle_id, float(fcd.start_offset), fcd.speed, fcd.active) for fcd in buffer],
            dtype=compound_dtype
        )

        # Validate the data
        for record in data:
            validate_data(record)  # Validate each record

        # Append data to the HDF5 file
        data_len = len(data)
        self.file["fcd"].resize((self.index + data_len,))
        self.file["fcd"][self.index:self.index + data_len] = data
        self.index += data_len
        self.file.flush()
        return data_len

    def close(self):
        self.file.close()
