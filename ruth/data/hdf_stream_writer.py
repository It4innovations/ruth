from datetime import datetime, timedelta
import os
from typing import List

import h5py
import numpy as np

from .map import Map

# Unix epoch for timezone-independent timestamp calculation.
_EPOCH = datetime(1970, 1, 1)

compound_dtype = np.dtype([
    ("timestamp", np.int64),       # timestamp in seconds
    ("node_from", np.int64),
    ("node_to", np.int64),
    ("segment_length", np.int32),
    ("vehicle_id", np.int64),
    ("start_offset_m", np.float32),
    ("speed_mps", np.float32),
    ("active", np.bool_),
])


def fcd_columns_to_array(datetime_seconds, vehicle_id, segment_id, offset_from_start,
                         vehicle_speed_mps, active, routing_map) -> np.ndarray:
    """Vectorised counterpart of fcd_records_to_array, straight from columns.

    The distributed advance already has these values as numeric arrays; building
    FCDRecord objects just to take them apart again costs an O(records) Python
    pass at each end. Everything here is whole-array work except one dict lookup
    per DISTINCT segment in the batch, which is far fewer than one per record.

    `segment_id` holds routing ids (Map.segment_id_to_int). node_from, node_to
    and the segment length are resolved through a cache keyed by routing id.
    """
    n = len(vehicle_id)
    if n == 0:
        return np.empty((0,), dtype=compound_dtype)

    unique_ids, inverse = np.unique(segment_id, return_inverse=True)
    node_from = np.empty(len(unique_ids), dtype=np.int64)
    node_to = np.empty(len(unique_ids), dtype=np.int64)
    length = np.empty(len(unique_ids), dtype=np.int32)
    for i, routing_id in enumerate(unique_ids):
        nf, nt, seg_len = _segment_row(routing_map, int(routing_id))
        node_from[i] = nf
        node_to[i] = nt
        length[i] = seg_len

    out = np.empty(n, dtype=compound_dtype)
    # int(...) on the record path truncates toward zero; astype does the same.
    out["timestamp"] = np.asarray(datetime_seconds, dtype=np.float64).astype(np.int64)
    out["node_from"] = node_from[inverse]
    out["node_to"] = node_to[inverse]
    out["segment_length"] = length[inverse]
    out["vehicle_id"] = np.asarray(vehicle_id, dtype=np.int64)
    out["start_offset_m"] = np.asarray(offset_from_start, dtype=np.float32)
    out["speed_mps"] = np.asarray(vehicle_speed_mps, dtype=np.float32)
    out["active"] = np.asarray(active, dtype=np.bool_)
    return out


def _segment_row(routing_map, routing_id: int):
    """routing_id -> (node_from, node_to, int(length)), memoised.

    The cache hangs off the MAP, not off this module. Routing ids are only
    meaningful relative to the map that assigned them, so a module-global cache
    would hand out another map's geometry the moment a process used a second map
    -- which a test suite does even though a run does not.
    """
    cache = getattr(routing_map, "_fcd_segment_rows", None)
    if cache is None:
        cache = {}
        routing_map._fcd_segment_rows = cache

    row = cache.get(routing_id)
    if row is None:
        node_from, node_to = routing_map.int_to_segment_id(routing_id)
        segment = routing_map.get_osm_segment(node_from, node_to)
        # int() not round(): matches int(fcd.segment.length) on the record path.
        row = (int(node_from), int(node_to), int(segment.length))
        cache[routing_id] = row
    return row


def fcd_records_to_array(buffer: List) -> np.ndarray:
    """Convert FCDRecord objects to the compact HDF5 compound-array representation.

    This function is separated from HDF5Writer so the simulator process can
    serialize compact NumPy arrays and let an async writer process own the HDF5
    file handle.
    """
    if not buffer:
        return np.empty((0,), dtype=compound_dtype)

    return np.array([
        (
            int((fcd.datetime - _EPOCH).total_seconds()),
            int(fcd.segment.node_from),
            int(fcd.segment.node_to),
            int(fcd.segment.length),
            int(fcd.vehicle_id),
            float(fcd.offset_from_start),
            float(fcd.vehicle_speed_mps),
            bool(fcd.active),
        )
        for fcd in buffer
    ], dtype=compound_dtype)


class HDF5Writer:
    def __init__(self, filename, dtype=None):
        if os.path.exists(filename):
            raise FileExistsError(f"The path {filename} exists.")

        self.file = h5py.File(filename, 'a')
        chunk_shape = (1024,)
        self.dataset = self.file.require_dataset(
            'fcd',
            shape=(0,),
            maxshape=(None,),
            dtype=compound_dtype,
            chunks=chunk_shape,
        )
        self.index = self.dataset.shape[0]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def save_computational_time(self, computational_time: float):
        if 'computational_time' not in self.file.attrs:
            self.file.attrs['computational_time'] = computational_time
        self.file.flush()

    def save_map_metadata(self, bbox, download_date, departure_time_iso: str, round_freq_s: float):
        """Save metadata without requiring a Map object.

        This is useful for the async writer process because sending the whole
        routing map through multiprocessing.Queue would be expensive and may not
        be picklable.
        """
        if 'bbox' not in self.file.attrs:
            self.file.attrs['bbox'] = tuple(bbox)

        if 'download_date' not in self.file.attrs:
            self.file.attrs['download_date'] = str(download_date)

        if 'departure_time' not in self.file.attrs:
            self.file.attrs['departure_time'] = departure_time_iso

        if 'round_freq_s' not in self.file.attrs:
            self.file.attrs['round_freq_s'] = float(round_freq_s)

        self.file.flush()

    def save_map(self, routing_map: Map, departure_time: datetime, round_freq: timedelta):
        self.save_map_metadata(
            bbox=tuple(routing_map.bbox.get_coords()),
            download_date=str(routing_map.download_date),
            departure_time_iso=departure_time.isoformat(),
            round_freq_s=float(round_freq.total_seconds()),
        )

    def append_array(self, data: np.ndarray):
        """Append an already-converted NumPy compound array to the HDF5 file."""
        data_len = len(data)
        if data_len == 0:
            return 0

        self.file['fcd'].resize((self.index + data_len,))
        self.file['fcd'][self.index:self.index + data_len] = data
        self.index += data_len
        self.file.flush()
        return data_len

    def append_file(self, buffer: List):
        """Backward-compatible path: convert FCDRecord objects and append."""
        return self.append_array(fcd_records_to_array(buffer))

    def close(self):
        try:
            self.file.close()
        except Exception:
            pass
