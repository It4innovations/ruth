import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import h5py

from ruth.data.hdf_stream_writer import HDF5Writer


class DummyBBox:
    def get_coords(self):
        return (0.0, 0.0, 1.0, 1.0)


class DummyMap:
    def __init__(self):
        self.bbox = DummyBBox()
        self.download_date = "2025-01-01"


class DummySegment:
    def __init__(self, node_from, node_to, length):
        self.node_from = node_from
        self.node_to = node_to
        self.length = length


class DummyFCD:
    def __init__(self, dt, node_from, node_to, length, vehicle_id, offset, speed, active):
        self.datetime = dt
        self.segment = DummySegment(node_from, node_to, length)
        self.vehicle_id = vehicle_id
        self.offset_from_start = offset
        self.vehicle_speed_mps = speed
        self.active = active


def test_hdf5_writer_append_and_reopen(tmp_path: Path):
    fpath = tmp_path / "fcd_test.h5"

    now = datetime.now(tz=timezone.utc)
    batch1 = [DummyFCD(now, 1, 2, 100, 11, 10.5, 5.0, True),
              DummyFCD(now, 3, 4, 200, 12, 20.0, 6.0, False)]

    batch2 = [DummyFCD(now + timedelta(seconds=1), 5, 6, 300, 21, 0.0, 4.5, True)]

    writer = HDF5Writer(str(fpath))
    writer.save_map(DummyMap(), departure_time=now, round_freq=timedelta(seconds=5))
    n1 = writer.append_file(batch1)
    assert n1 == len(batch1)
    assert writer.index == len(batch1)
    writer.close()

    # reopen and append second batch
    writer2 = HDF5Writer(str(fpath))
    assert writer2.index == len(batch1)
    n2 = writer2.append_file(batch2)
    assert n2 == len(batch2)
    assert writer2.index == len(batch1) + len(batch2)

    # verify attributes and dataset contents using h5py directly
    writer2.file.flush()
    with h5py.File(str(fpath), 'r') as f:
        assert 'fcd' in f
        ds = f['fcd']
        assert ds.shape[0] == len(batch1) + len(batch2)
        assert 'bbox' in f.attrs
        assert 'download_date' in f.attrs
        assert 'departure_time' in f.attrs
        assert 'round_freq_s' in f.attrs
        data = ds[:]
        vid_values = [int(r['vehicle_id']) for r in data]
        assert 11 in vid_values and 12 in vid_values and 21 in vid_values

    writer2.close()

    if os.path.exists(str(fpath)):
        os.remove(str(fpath))

