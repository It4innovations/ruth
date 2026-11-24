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


# ---------------------------------------------------------------------------
# Task digest: the other half of the comparison.
#
# The trajectory digest hashes where the vehicles ARE. It deliberately does not
# hash routes, segment geometry or closures, because those are not part of "the
# vehicle is in the same position". That blind spot matters when two runs agree
# on every position at step N and disagree at step N+1: the kernel is a pure
# function of (task, GlobalView), so if the positions going in were identical,
# either a task field differed -- a route that was rewritten differently, a
# closure that landed on a different segment -- or the GlobalView did.
#
# Hashing the task array splits those two cases with one number per step:
#
#   * task digests already differ at the first diverging step -> the inputs
#     diverged, and the dump names the field. Nothing to look for in the LoS.
#   * task digests are equal and the positions still differ -> every scalar
#     input was identical, so the only remaining input is the GlobalView.
#
# Rows are hashed in vehicle-id order for the same reason the trajectory digest
# is: the digest must depend on the SET of tasks, not on how they were split
# across free/queue buffers or distributed across ranks.

# Hashed field by field rather than with one tobytes() over the row: the dtype
# carries padding between the SegmentDesc members, and padding is not part of
# the kernel's input. Reading it would make the digest depend on whether the
# buffer came from np.zeros or from C++.
_TASK_FIELDS = (
    "vehicle_id", "start_index", "start_offset_m", "time_offset_us",
    "frequency_us", "fcd_period_us",
    "seg0_id", "seg0_len", "seg0_max_kph", "seg0_lanes", "seg0_valid", "seg0_closed",
    "seg1_id", "seg1_len", "seg1_max_kph", "seg1_lanes", "seg1_valid", "seg1_closed",
)


def _tasks_in_id_order(task_arrays):
    """The step's task buffers concatenated and sorted by vehicle id, or None."""
    present = [arr for arr in task_arrays if arr is not None and len(arr) > 0]
    if not present:
        return None
    combined = np.concatenate(present) if len(present) > 1 else present[0]
    order = np.argsort(combined["vehicle_id"], kind="stable")
    return combined[order]


def task_digest(task_arrays) -> str:
    """Hex digest of the step's AdvanceTask inputs, order-independent."""
    tasks = _tasks_in_id_order(task_arrays)
    hasher = blake2b(digest_size=_DIGEST_SIZE)
    if tasks is None or len(tasks) == 0:
        return hasher.hexdigest()
    for name in _TASK_FIELDS:
        hasher.update(np.ascontiguousarray(tasks[name]).tobytes())
    return hasher.hexdigest()


def dump_tasks(task_arrays, path: str) -> int:
    """Write the step's tasks to `path` as CSV. Returns the row count.

    Same 17-significant-digit floats as dump_trajectory, so two dumps diff
    exactly and the first differing column names the field that diverged.
    """
    tasks = _tasks_in_id_order(task_arrays)
    if tasks is None:
        return 0

    float_fields = {"start_offset_m", "seg0_len", "seg0_max_kph",
                    "seg1_len", "seg1_max_kph"}
    with open(path, "w") as handle:
        handle.write(",".join(_TASK_FIELDS) + "\n")
        for row in tasks:
            handle.write(",".join(
                ("%.17g" % row[name]) if name in float_fields else ("%d" % row[name])
                for name in _TASK_FIELDS) + "\n")
    return len(tasks)
