"""
Production accessors for the C++ VehicleStore.

Separate from store_bridge, which is population upkeep and the shadow checks:
these four are on the per-step hot path and nothing here verifies anything.

The store is the source of truth for the step's selection. Walking every Vehicle
in Python to find the few thousand that move cost ~384 ms per step at 936k
vehicles, and ~68 ms more to find the next offset -- neither scaling with how
many vehicles were actually active. The store answers both from its own columns,
so the Python work becomes proportional to the answer instead of the population.
"""

import numpy as np

import ruthlib


def apply_delta_columns(vehicle_id, new_start_index, new_offset_m, new_time_offset_us,
                        moved, changed_segment, reached_end) -> dict:
    """Apply already-ordered delta columns to the store.

    Keeping the store's movement columns current is what makes selection
    possible: the alternative is a Python pass over every vehicle each step,
    which costs more than the selection saves. The caller has already permuted
    these columns into replay order for its own loop, so there is no dict and no
    second pass here.

    Order still matters -- it is the order the queue events come back in.
    """
    return ruthlib.advance_store_apply_deltas(
        np.ascontiguousarray(vehicle_id, dtype=np.int32),
        np.ascontiguousarray(new_start_index, dtype=np.int32),
        np.ascontiguousarray(new_offset_m, dtype=np.float64),
        np.ascontiguousarray(new_time_offset_us, dtype=np.int64),
        np.ascontiguousarray(moved, dtype=np.int32),
        np.ascontiguousarray(changed_segment, dtype=np.int32),
        np.ascontiguousarray(reached_end, dtype=np.int32))


def advance_waiting(vehicles) -> None:
    """Advance the store's clock for vehicles that waited this step."""
    if not vehicles:
        return
    ids = np.fromiter((v.id for v in vehicles), dtype=np.int32, count=len(vehicles))
    ruthlib.advance_store_advance_waiting(ids)


def select_within_offset(offset_seconds: int, freq_seconds: int) -> np.ndarray:
    """Vehicle ids that move at this offset, in slot order.

    Same rule as Simulation.is_vehicle_within_offset, Python's
    round-half-to-even included; see VehicleStore::select_within_offset.
    """
    return ruthlib.advance_store_select_within_offset(int(offset_seconds),
                                                      int(freq_seconds))


def at_segment_end(vehicle_ids) -> np.ndarray:
    """Of those ids, the ones sitting exactly at the end of their segment."""
    return ruthlib.advance_store_at_segment_end(
        np.ascontiguousarray(vehicle_ids, dtype=np.int32))


def min_active_time_offset_us():
    """Smallest non-zero time offset among active vehicles, or None.

    Mirrors compute_current_offset(), filter(None) quirk included -- a vehicle
    sitting at exactly zero does not participate.
    """
    return ruthlib.advance_store_min_active_time_offset_us()
