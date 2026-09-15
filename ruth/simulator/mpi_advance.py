"""
Distributed vehicle-advance step (MPI/ACE backend).

Drop-in replacement for route.advance_vehicles_with_queues that offloads the
expensive per-vehicle movement + FCD generation to ruthlib workloads spread
across MPI ranks, then all-gathers the resulting FCDs into every rank's
GlobalView replica (see binding/src/advance_*).

The master keeps ownership of the Vehicle objects and the QueuesManager: it
builds the per-vehicle tasks, and after the C++ step it applies the returned
movement deltas, replays the queue bookkeeping in canonical order (reusing the
existing QueuesManager), and materializes the FCD history. This mirrors
route.advance_vehicles_with_queues exactly; only the kinematics run in C++.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np

import ruthlib

from ..data.segment import LengthMeters, Segment, SpeedMps
from ..data.map import Map
from ..vehicle import Vehicle
from .queues import QueuesManager
from .simulation import FCDRecord

logger = logging.getLogger(__name__)

# Debug builds (CMAKE_BUILD_TYPE Debug/RelWithDebInfo) compile ruthlib with the
# FCD-sort flag on; when set, the master reorders each step's gathered FCDs into
# serial's write order (free, then queues FIFO, then waiting) so the FCD history
# is byte-identical to a serial run. Off in Release (skips the reorder).
try:
    _DEBUG_SORT_FCDS = bool(ruthlib.advance_debug_sort_enabled())
except AttributeError:
    _DEBUG_SORT_FCDS = False

# Time is carried as integer microseconds end-to-end to match the serial path's
# timedelta arithmetic exactly. Using float seconds accumulates sub-second drift
# that the LoS tolerance window amplifies into divergent trajectories.
_EPOCH = datetime(1970, 1, 1)
_ONE_US = timedelta(microseconds=1)


def _td_to_us(td: timedelta) -> int:
    """Exact integer microseconds of a timedelta (matches C++ int64 µs time)."""
    return td // _ONE_US


def _datetime_to_epoch_us(dt: datetime) -> int:
    """Exact integer epoch microseconds (naive UTC, as datetime_to_timestamp)."""
    return (dt - _EPOCH) // _ONE_US

# numpy structured dtype matching the C++ ruthlib::AdvanceTask layout exactly.
# Offsets/itemsize are explicit and validated against the C++ struct at import.
ADVANCE_TASK_DTYPE = np.dtype(dict(
    names=[
        "vehicle_id", "start_index", "start_offset_m", "time_offset_us",
        "frequency_us", "fcd_period_us",
        "seg0_id", "seg0_len", "seg0_max_kph", "seg0_lanes", "seg0_valid", "seg0_closed",
        "seg1_id", "seg1_len", "seg1_max_kph", "seg1_lanes", "seg1_valid", "seg1_closed",
    ],
    formats=[
        "i4", "i4", "f8", "i8",
        "i8", "i8",
        "i8", "f8", "f8", "i4", "i4", "i4",
        "i8", "f8", "f8", "i4", "i4", "i4",
    ],
    # Offsets match the C++ AdvanceTask / SegmentDesc layout (each SegmentDesc is
    # padded to 40 bytes). The named fields skip the 4-byte trailing pad of each
    # SegmentDesc; itemsize covers the whole padded struct.
    offsets=[
        0, 4, 8, 16,
        24, 32,
        40, 48, 56, 64, 68, 72,
        80, 88, 96, 104, 108, 112,
    ],
    itemsize=120,
))

_expected = ruthlib.advance_task_itemsize()
if _expected != ADVANCE_TASK_DTYPE.itemsize:
    raise ImportError(
        f"ADVANCE_TASK_DTYPE itemsize ({ADVANCE_TASK_DTYPE.itemsize}) does not match "
        f"the C++ AdvanceTask size ({_expected}); the struct layout has drifted."
    )


def _build_task_array(vehicles: List[Vehicle], routing_map: Map) -> np.ndarray:
    """One ADVANCE_TASK_DTYPE row per vehicle + its ≤2 upcoming segments.

    Filled column-wise: whole-column assignment from list comprehensions
    replaces the per-row structured __setitem__ (14 field resolutions per
    vehicle per step), which profiling put at ~1.5% of wall on its own.
    """
    n = len(vehicles)
    arr = np.zeros(n, dtype=ADVANCE_TASK_DTYPE)
    if n == 0:
        return arr

    seg_int = routing_map.segment_id_to_int
    cur_max = routing_map.get_current_max_speed
    seg_pairs = [
        routing_map.osm_route_to_py_segments(
            v.osm_route[v.start_index:v.start_index + 3])
        for v in vehicles
    ]

    arr["vehicle_id"] = [v.id for v in vehicles]
    arr["start_index"] = [v.start_index for v in vehicles]
    arr["start_offset_m"] = [v.start_distance_offset for v in vehicles]
    arr["time_offset_us"] = [_td_to_us(v.time_offset) for v in vehicles]
    arr["frequency_us"] = [_td_to_us(v.frequency) for v in vehicles]
    arr["fcd_period_us"] = [_td_to_us(v.fcd_sampling_period) for v in vehicles]

    seg0 = [segs[0] for segs in seg_pairs]
    arr["seg0_id"] = [seg_int(s.id) for s in seg0]
    arr["seg0_len"] = [s.length for s in seg0]
    arr["seg0_max_kph"] = [s.max_allowed_speed_kph for s in seg0]  # original limit (for speed*LoS)
    arr["seg0_lanes"] = [s.lanes for s in seg0]
    arr["seg0_valid"] = 1
    # is_closed uses the CURRENT max speed (closures live in the current network),
    # matching Vehicle.has_next_segment_closed / Map.get_current_max_speed.
    arr["seg0_closed"] = [1 if cur_max(s.node_from, s.node_to) == 0 else 0 for s in seg0]

    idx1 = [i for i, segs in enumerate(seg_pairs) if len(segs) > 1]
    if idx1:
        seg1 = [seg_pairs[i][1] for i in idx1]
        arr["seg1_id"][idx1] = [seg_int(s.id) for s in seg1]
        arr["seg1_len"][idx1] = [s.length for s in seg1]
        arr["seg1_max_kph"][idx1] = [s.max_allowed_speed_kph for s in seg1]
        arr["seg1_lanes"][idx1] = [s.lanes for s in seg1]
        arr["seg1_valid"][idx1] = 1
        arr["seg1_closed"][idx1] = [1 if cur_max(s.node_from, s.node_to) == 0 else 0 for s in seg1]
    # rows without a second segment: seg1_valid / seg1_closed stay 0 (np.zeros init)

    return arr


def _split_vehicles(vehicles: List[Vehicle], queues_manager: QueuesManager
                    ) -> Tuple[List[Vehicle], List[List[Vehicle]], List[Vehicle]]:
    """
    Partition the vehicles exactly as route.advance_vehicles_with_queues does:

      * free            - not in a queue; order-independent.
      * queue_groups    - per queue, the FIFO prefix of selected vehicles (the
                          loop stops at the first vehicle not selected this step,
                          route.py:305). Each group is one whole QueueWorkload.
      * waiting         - selected+queued vehicles that fall *after* such a gap;
                          they wait this step (route.py:324) and are handled by
                          the master (their stationary FCDs are injected).
    """
    in_queue: Dict[int, Vehicle] = {}
    free: List[Vehicle] = []
    for vehicle in vehicles:
        queue = queues_manager.queues[(vehicle.current_node, vehicle.next_node)]
        if vehicle.id in queue:
            in_queue[vehicle.id] = vehicle
        else:
            free.append(vehicle)

    queue_groups: List[List[Vehicle]] = []
    processed = set()
    for _, queue in queues_manager.queues.items():
        group: List[Vehicle] = []
        for vehicle_id in queue:
            if vehicle_id not in in_queue:
                break
            group.append(in_queue[vehicle_id])
            processed.add(vehicle_id)
        if group:
            queue_groups.append(group)

    waiting = [v for vid, v in in_queue.items() if vid not in processed]
    return free, queue_groups, waiting


def _inject_waiting_fcds(waiting: List[Vehicle], routing_map: Map, departure_epoch_us: int):
    """Stationary FCDs for after-gap waiting vehicles, added to the master's local
    accumulator so they join the all-gather (route.advance_waiting_vehicle).
    The datetime is derived from integer µs (then / 1e6) to match the kernel."""
    if not waiting:
        return
    n = len(waiting)
    dt = np.empty(n, dtype=np.float64)
    vid = np.empty(n, dtype=np.int32)
    sid = np.empty(n, dtype=np.int64)
    off = np.empty(n, dtype=np.float64)
    spd = np.zeros(n, dtype=np.float64)  # stationary
    for i, v in enumerate(waiting):
        idx = v.start_index
        seg0 = routing_map.osm_route_to_py_segments(v.osm_route[idx:idx + 3])[0]
        end_us = departure_epoch_us + _td_to_us(v.time_offset) + _td_to_us(v.frequency)
        dt[i] = end_us / 1e6
        vid[i] = v.id
        sid[i] = routing_map.segment_id_to_int(seg0.id)
        off[i] = v.start_distance_offset
    ruthlib.advance_add_local_fcds(dt, vid, sid, off, spd)


def _apply_delta(vehicle: Vehicle, d, routing_map: Map, queues_manager: QueuesManager) -> bool:
    """Apply one movement delta and replay route.advance_vehicle's queue bookkeeping.
    Returns whether the vehicle moved."""
    old_index = vehicle.start_index

    vehicle.start_index = int(d["new_start_index"])
    vehicle.start_distance_offset = LengthMeters(float(d["new_offset_m"]))
    vehicle.time_offset = timedelta(microseconds=int(d["new_time_offset_us"]))

    moved = bool(d["moved"])

    # Left the previous segment's queue (was at its end, index advanced).
    if d["changed_segment"]:
        node_from = vehicle.osm_route[old_index]
        node_to = vehicle.osm_route[old_index + 1]
        queues_manager.remove_vehicle(vehicle, node_from, node_to)

    # Reached the end of the current segment on a moving step: either arrive or
    # queue up. (Gating on `moved` keeps blockers/waiters no-ops, matching
    # route.py's `segment_pos_old != vehicle.segment_position`.)
    if d["reached_end"] and moved:
        if vehicle.next_node == vehicle.dest_node:
            vehicle.active = False
        else:
            queues_manager.add_to_queue(vehicle)

    return moved


def _build_history(fcd_arrays: dict, moved_by_id: Dict[int, Vehicle],
                   routing_map: Map) -> List[FCDRecord]:
    """Materialize FCDRecords from the gathered numeric FCDs, annotating status
    and active. Per vehicle, the latest-datetime sample carries the vehicle's
    final `active`; the rest are active (matching route.generate_fcds)."""
    times = fcd_arrays["datetime_seconds"]
    vids = fcd_arrays["vehicle_id"]
    sids = fcd_arrays["segment_id"]
    offs = fcd_arrays["offset_from_start"]
    spds = fcd_arrays["vehicle_speed_mps"]
    n = len(vids)

    # Index of each vehicle's latest FCD (its end-of-movement sample).
    last_index_by_vid: Dict[int, int] = {}
    for i in range(n):
        vid = int(vids[i])
        prev = last_index_by_vid.get(vid)
        if prev is None or times[i] > times[prev]:
            last_index_by_vid[vid] = i

    seg_cache: Dict[int, Segment] = {}

    def segment_of(sid_int: int) -> Segment:
        seg = seg_cache.get(sid_int)
        if seg is None:
            node_from, node_to = routing_map.int_to_segment_id(sid_int)
            seg = routing_map.get_osm_segment(node_from, node_to)
            seg_cache[sid_int] = seg
        return seg

    fcds: List[FCDRecord] = []
    for i in range(n):
        vid = int(vids[i])
        vehicle = moved_by_id[vid]
        is_last = (last_index_by_vid[vid] == i)
        active = vehicle.active if is_last else True
        fcds.append(FCDRecord(
            # times[i] is absolute epoch seconds (= µs/1e6); _EPOCH + that
            # reproduces the serial fcd.datetime (= departure_time + offset).
            datetime=_EPOCH + timedelta(seconds=float(times[i])),
            vehicle_id=vid,
            segment=segment_of(int(sids[i])),
            offset_from_start=LengthMeters(float(offs[i])),
            vehicle_speed_mps=SpeedMps(float(spds[i])),
            status=vehicle.status,
            active=active,
        ))
    return fcds


def _reorder_fcds_to_serial(fcd_arrays: dict, free: List[Vehicle],
                            queue_groups: List[List[Vehicle]], waiting: List[Vehicle]) -> dict:
    """Reorder the gathered FCDs into serial's exact write order: free vehicles
    (in split order), then each queue group FIFO, then waiting — and by datetime
    within each vehicle. Debug aid only; makes the history byte-identical to serial.
    The workloads and the serial path are untouched — this is pure master-side
    reordering, valid because the per-step FCD *set* is identical to serial's."""
    order: Dict[int, int] = {}
    k = 0
    for vehicle in free:
        order[vehicle.id] = k; k += 1
    for group in queue_groups:
        for vehicle in group:
            order[vehicle.id] = k; k += 1
    for vehicle in waiting:
        order[vehicle.id] = k; k += 1

    vids = fcd_arrays["vehicle_id"]
    times = fcd_arrays["datetime_seconds"]
    proc = np.fromiter((order.get(int(v), 1 << 31) for v in vids),
                       dtype=np.int64, count=len(vids))
    perm = np.lexsort((times, proc))   # primary: processing order; secondary: datetime
    return {key: np.asarray(arr)[perm] for key, arr in fcd_arrays.items()}


def advance_vehicles_mpi(vehicles_to_be_moved: List[Vehicle], departure_time,
                         routing_map: Map, queues_manager: QueuesManager,
                         los_vehicles_tolerance, drop_before_time=None
                         ) -> Tuple[List[FCDRecord], bool]:
    """Distributed equivalent of route.advance_vehicles_with_queues.

    Note: LoS is evaluated against the C++ GlobalView replicas (kept in sync by
    the all-gather), not the Python global_view — so this does not take a gv_db.
    """
    departure_epoch_us = _datetime_to_epoch_us(departure_time)
    los_tolerance_s = los_vehicles_tolerance.total_seconds()

    free, queue_groups, waiting = _split_vehicles(vehicles_to_be_moved, queues_manager)

    # Build task buffers (structured dtype) + queue CSR offsets.
    free_arr = _build_task_array(free, routing_map)
    flat_queue_vehicles = [v for group in queue_groups for v in group]
    queue_arr = _build_task_array(flat_queue_vehicles, routing_map)
    queue_offsets = np.zeros(len(queue_groups) + 1, dtype=np.int32)
    acc = 0
    for i, group in enumerate(queue_groups):
        queue_offsets[i] = acc
        acc += len(group)
    queue_offsets[len(queue_groups)] = acc

    # Prune every GV replica BEFORE do_advance reads it, at the current offset —
    # this reproduces serial's read-time GV (serial prunes at the end of the
    # previous step with the current offset). Doing this in the sync (after the
    # read) pruned one round too late and diverged the LoS. Threshold is in the
    # GV's double-second units, derived from µs to match the FCD datetimes.
    drop_before = (_datetime_to_epoch_us(drop_before_time) / 1e6
                   if drop_before_time is not None else float("-inf"))
    ruthlib.advance_drop_old(drop_before)

    # Phase 1: scatter + evaluate across ranks (reads the just-pruned GV).
    ruthlib.do_advance(free_arr, queue_arr, queue_offsets, departure_epoch_us, los_tolerance_s)

    # After-gap waiting vehicles: inject their stationary FCDs before the sync.
    _inject_waiting_fcds(waiting, routing_map, departure_epoch_us)

    # Phase 2: all-gather the step's FCDs into every GV replica (no pruning here).
    ruthlib.sync_advance()

    deltas = ruthlib.get_advance_deltas()
    fcd_arrays = ruthlib.get_advance_fcds()

    # Index deltas by vehicle id, then apply in canonical order (free first, then
    # each queue group in FIFO order) so the queue batch_update is deterministic.
    delta_by_id = {int(deltas["vehicle_id"][i]): {
        "new_start_index": deltas["new_start_index"][i],
        "new_offset_m": deltas["new_offset_m"][i],
        "new_time_offset_us": deltas["new_time_offset_us"][i],
        "moved": deltas["moved"][i],
        "changed_segment": deltas["changed_segment"][i],
        "reached_end": deltas["reached_end"][i],
    } for i in range(len(deltas["vehicle_id"]))}

    vehicles_moved = False
    for vehicle in free:
        vehicles_moved |= _apply_delta(vehicle, delta_by_id[vehicle.id], routing_map, queues_manager)
    for group in queue_groups:
        for vehicle in group:
            vehicles_moved |= _apply_delta(vehicle, delta_by_id[vehicle.id], routing_map, queues_manager)
    # After-gap waiting: advance time only, no movement, no queue op.
    for vehicle in waiting:
        vehicle.time_offset += vehicle.frequency

    queues_manager.batch_update()

    # Debug only: reorder gathered FCDs into serial's write order so the FCD
    # history is byte-identical (content is already identical either way).
    if _DEBUG_SORT_FCDS:
        fcd_arrays = _reorder_fcds_to_serial(fcd_arrays, free, queue_groups, waiting)

    moved_by_id = {v.id: v for v in vehicles_to_be_moved}
    fcds = _build_history(fcd_arrays, moved_by_id, routing_map)

    return fcds, vehicles_moved
