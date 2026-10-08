"""
Per-step trajectory digest: the compliance instrument for the distributed advance.

The acceptance criterion for the distributed advance is that every vehicle is in
the same position at every timestep as in a serial run. FCD log row order is
explicitly allowed to differ, so diffing the HDF5 history cannot express that
criterion: it reports differences that do not matter and buries the one that
does. This module hashes the movement state instead -- one digest per step,
logged beside the step timers -- so comparing two runs is comparing a column of
hex strings, and the first differing row names the first diverging step.

Two properties make the digest trustworthy:

  * Vehicles are hashed in vehicle-id order, so the digest depends only on the
    SET of vehicle states. It is blind to list order and to how the work was
    distributed, which is exactly the invariance we are trying to verify.
  * Floats are hashed as their exact IEEE-754 bytes and times as integer
    microseconds, so a last-bit divergence shows up instead of being rounded
    away. A digest that matched "closely enough" would be worthless here.

Enabled by config.json `ace.trajectory-digest`; off by default because it costs
one O(N) pass per step. On a mismatch, set `ace.trajectory-dump-step` to the
first diverging step to get the full per-vehicle state as CSV and find the
vehicle that moved differently.
"""

from datetime import timedelta
from hashlib import blake2b
from typing import Sequence, Tuple

import numpy as np

# Exact integer microseconds, matching the int64 microsecond time the C++ advance
# kernel carries (see mpi_advance._td_to_us). Float seconds would reintroduce the
# rounding this digest exists to detect.
_ONE_US = timedelta(microseconds=1)

# 16 bytes is plenty: this distinguishes runs, it does not authenticate them.
_DIGEST_SIZE = 16

_COLUMN_NAMES = ("vehicle_id", "start_index", "start_distance_offset_m",
                 "time_offset_us", "active")


def movement_columns(vehicles: Sequence) -> Tuple[np.ndarray, ...]:
    """The movement state of `vehicles` as columns, sorted by vehicle id.

    This is the full set of per-vehicle state the advance step may change:
    anything outside it (status, route selection, travel times) is not part of
    "the vehicle is in the same position".
    """
    n = len(vehicles)
    vehicle_id = np.empty(n, dtype=np.int64)
    start_index = np.empty(n, dtype=np.int64)
    offset_m = np.empty(n, dtype=np.float64)
    time_offset_us = np.empty(n, dtype=np.int64)
    active = np.empty(n, dtype=np.uint8)

    for i, vehicle in enumerate(vehicles):
        vehicle_id[i] = vehicle.id
        start_index[i] = vehicle.start_index
        offset_m[i] = vehicle.start_distance_offset
        time_offset_us[i] = vehicle.time_offset // _ONE_US
        active[i] = 1 if vehicle.active else 0

    # Stable sort so a duplicated id (which would be a bug elsewhere) still gives
    # a reproducible digest rather than an arbitrary one.
    order = np.argsort(vehicle_id, kind="stable")
    return (vehicle_id[order], start_index[order], offset_m[order],
            time_offset_us[order], active[order])


def trajectory_digest(vehicles: Sequence) -> str:
    """Hex digest of the movement state of `vehicles`, order-independent."""
    hasher = blake2b(digest_size=_DIGEST_SIZE)
    if len(vehicles) == 0:
        return hasher.hexdigest()

    for column in movement_columns(vehicles):
        # ascontiguousarray: the fancy-indexed columns are already contiguous,
        # but tobytes() on a non-contiguous view would silently hash a copy in a
        # different layout, so pin it rather than rely on that.
        hasher.update(np.ascontiguousarray(column).tobytes())
    return hasher.hexdigest()


def dump_trajectory(vehicles: Sequence, path: str) -> int:
    """Write the full movement state to `path` as CSV. Returns the row count.

    Offsets are written with 17 significant digits, which round-trips a double
    exactly, so two dumps can be diffed with plain `diff` without a near-equal
    comparison hiding the divergence.
    """
    columns = movement_columns(vehicles)
    vehicle_id, start_index, offset_m, time_offset_us, active = columns

    with open(path, "w") as handle:
        handle.write(",".join(_COLUMN_NAMES) + "\n")
        for i in range(len(vehicle_id)):
            handle.write("%d,%d,%.17g,%d,%d\n" % (
                vehicle_id[i], start_index[i], offset_m[i],
                time_offset_us[i], active[i]))
    return len(vehicle_id)
