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
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

import ruthlib

from ..data.segment import LengthMeters, Segment, SpeedMps
from ..data.map import Map
from ..vehicle import Vehicle
from .queues import QueuesManager
from .simulation import FCDColumns, FCDRecord

logger = logging.getLogger(__name__)

# Whether each step's gathered FCDs are reordered into serial's write order is a
# RUNTIME choice (config.json ace.deterministic-order), passed in per call by the
# caller. It used to be a compile-time flag (RUTH_ADVANCE_SORT_FCDS, exposed as
# ruthlib.advance_debug_sort_enabled()), which tied reproducibility to the build
# flavour and could not be flipped to validate a release build; that flag and its
# accessor are removed.
#
# This affects the ORDER of the FCD history rows only, NOT vehicle positions.
# Positions do not depend on FCD order: the LoS query counts a set of distinct
# vehicle ids, drop_old works on whole batches, and GlobalView::get_segment_speed
# sums in vehicle-id order. Turn this on to diff the HDF5 history against a
# serial baseline; leave it off otherwise and compare the per-step trajectory
# digests instead (trajectory_digest.py), which is the real compliance check.

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


# Task-digest hand-off (ace.trajectory-digest). The step loop knows the step
# number; this module knows the tasks, and they are gone again by the time the
# loop regains control. Rather than widen advance_vehicles_mpi's return type for
# a diagnostic, the loop arms these before the call and reads last_task_digest
# after it. Both are reset every step, so a stale digest cannot be logged.
task_digest_enabled = False
task_dump_path = None
last_task_digest = None


def _emit_task_digest(free_arr, queue_arr) -> None:
    """Hash (and optionally dump) the tasks the kernel is about to be given."""
    global last_task_digest
    last_task_digest = None
    if not task_digest_enabled:
        return

    from .trajectory_digest import dump_tasks, task_digest
    arrays = (free_arr, queue_arr)
    last_task_digest = task_digest(arrays)
    if task_dump_path is not None:
        rows = dump_tasks(arrays, task_dump_path)
        logger.info("task_digest: wrote %s task rows to %s", rows, task_dump_path)


def _fill_task(row, vehicle: Vehicle, routing_map: Map):
    """Fill one ADVANCE_TASK_DTYPE row from a vehicle + its ≤2 upcoming segments."""
    idx = vehicle.start_index
    segs = routing_map.osm_route_to_py_segments(vehicle.osm_route[idx:idx + 3])
    seg0 = segs[0]

    row["vehicle_id"] = vehicle.id
    row["start_index"] = vehicle.start_index
    row["start_offset_m"] = vehicle.start_distance_offset
    row["time_offset_us"] = _td_to_us(vehicle.time_offset)
    row["frequency_us"] = _td_to_us(vehicle.frequency)
    row["fcd_period_us"] = _td_to_us(vehicle.fcd_sampling_period)

    row["seg0_id"] = routing_map.segment_id_to_int(seg0.id)
    row["seg0_len"] = seg0.length
    row["seg0_max_kph"] = seg0.max_allowed_speed_kph  # original limit (for speed*LoS)
    row["seg0_lanes"] = seg0.lanes
    row["seg0_valid"] = 1
    # is_closed uses the CURRENT max speed (closures live in the current network),
    # matching Vehicle.has_next_segment_closed / Map.get_current_max_speed.
    row["seg0_closed"] = 1 if routing_map.get_current_max_speed(seg0.node_from, seg0.node_to) == 0 else 0

    if len(segs) > 1:
        seg1 = segs[1]
        row["seg1_id"] = routing_map.segment_id_to_int(seg1.id)
        row["seg1_len"] = seg1.length
        row["seg1_max_kph"] = seg1.max_allowed_speed_kph
        row["seg1_lanes"] = seg1.lanes
        row["seg1_valid"] = 1
        row["seg1_closed"] = 1 if routing_map.get_current_max_speed(seg1.node_from, seg1.node_to) == 0 else 0
    # else: seg1_valid / seg1_closed stay 0 (np.zeros init)


def _build_task_array(vehicles: List[Vehicle], routing_map: Map) -> np.ndarray:
    """Reference task builder: one Python row per vehicle.

    Superseded as the production path by the C++ VehicleStore (see
    store_bridge.build_tasks); this measured ~8.4 us/vehicle, dominated by the
    networkx edge lookups in _fill_task. It is kept, not deleted, because it is
    what `ace.store-shadow-check` compares the store against. Nothing else calls
    it -- it is a reference implementation, not a fallback.
    """
    arr = np.zeros(len(vehicles), dtype=ADVANCE_TASK_DTYPE)
    for i, vehicle in enumerate(vehicles):
        _fill_task(arr[i], vehicle, routing_map)
    return arr


def _split_vehicles(vehicles: List[Vehicle], queues_manager: QueuesManager
                    ) -> Tuple[List[Vehicle], List[List[Vehicle]], List[Vehicle]]:
    """
    Partition the vehicles exactly as route.advance_vehicles_with_queues does:

      * free            - not in a queue; order-independent.
      * queue_groups    - per queue, the FIFO prefix of selected vehicles (the
                          loop stops at the first vehicle not selected this step,
                          route.py:305). Groups are packed into batched
                          workloads but a group is never split across them.
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


def _build_fcd_columns(fcd_arrays: dict, moved_by_id: Dict[int, Vehicle]) -> FCDColumns:
    """The step's FCDs as columns, with `active` resolved per record.

    Replaces _build_history on the production path. The only thing the records
    added was `active` (and `status`, which nothing reads): per vehicle the
    latest sample carries that vehicle's final active flag and the rest are
    active, matching route.generate_fcds.

    The per-vehicle work here is O(distinct vehicles), not O(records): one
    lexsort finds each vehicle's last sample, and only those are looked up.
    """
    vehicle_id = np.asarray(fcd_arrays["vehicle_id"])
    n = len(vehicle_id)

    active = np.ones(n, dtype=np.bool_)
    if n:
        times = np.asarray(fcd_arrays["datetime_seconds"])
        # Sort by vehicle then time; the last row of each vehicle's run is its
        # latest sample. _build_history used a strict > when scanning, so on an
        # exact tie it kept the first -- generate_fcds emits strictly increasing
        # timestamps per vehicle, so no tie can arise either way.
        order = np.lexsort((times, vehicle_id))
        sorted_vehicles = vehicle_id[order]
        is_last = np.empty(n, dtype=bool)
        is_last[:-1] = sorted_vehicles[:-1] != sorted_vehicles[1:]
        is_last[-1] = True
        last_index = order[is_last]
        active[last_index] = np.fromiter(
            (moved_by_id[int(vehicle_id[i])].active for i in last_index),
            dtype=np.bool_, count=len(last_index))

    return FCDColumns(
        datetime_seconds=np.asarray(fcd_arrays["datetime_seconds"]),
        vehicle_id=vehicle_id,
        segment_id=np.asarray(fcd_arrays["segment_id"]),
        offset_from_start=np.asarray(fcd_arrays["offset_from_start"]),
        vehicle_speed_mps=np.asarray(fcd_arrays["vehicle_speed_mps"]),
        active=active,
    )


def _replay_deltas(replay_order: List[Vehicle], deltas: dict,
                   queues_manager: QueuesManager) -> Tuple[bool, Optional[dict]]:
    """Apply the step's deltas to `replay_order`, in that order.

    Returns (vehicles_moved, store_queue_events). The second is the queue
    operations the store staged while applying the same deltas, always a valid
    (possibly empty) event set. It has to come back out: applying the deltas is NOT idempotent, so whoever wants those
    events must take these rather than call the store a second time.

    Replaces a per-vehicle _apply_delta loop, with the per-vehicle
    intermediates removed. Three of them mattered:

      * the deltas arrive in gather order, so the old code built a dict of dicts
        keyed by vehicle id -- N dicts and seven numpy scalar reads each. One
        vectorised permutation puts the columns in replay order instead, and
        .tolist() hands the loop native scalars;
      * it was a function call per vehicle; and
      * Vehicle overrides __setattr__ to cache _frequency_seconds whenever
        `frequency` is written. Replay never writes `frequency`, so every
        movement write paid a Python-level dispatch and a string compare for
        nothing. Writing through __dict__ skips it and is otherwise identical --
        the override's only side effect is the one for `frequency`.

    Queue bookkeeping is unchanged and still goes through QueuesManager.
    """
    from . import store_select

    n = len(replay_order)
    if n == 0:
        # Still go through the store, so the caller gets a properly shaped empty
        # event set instead of None. The shadow check indexes into it without
        # checking, and a step where every selected vehicle is waiting reaches
        # here with nothing to replay.
        empty_i32 = np.empty(0, dtype=np.int32)
        return False, store_select.apply_delta_columns(
            empty_i32, empty_i32, np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int64), empty_i32, empty_i32, empty_i32)

    # Put the columns in replay order. searchsorted over the sorted gather order
    # is O(n log n) of vectorised work, against O(n) dict inserts before.
    gathered_ids = np.asarray(deltas["vehicle_id"])
    wanted_ids = np.fromiter((v.id for v in replay_order), dtype=gathered_ids.dtype,
                             count=n)
    sorter = np.argsort(gathered_ids)
    position = sorter[np.searchsorted(gathered_ids, wanted_ids, sorter=sorter)]

    if not np.array_equal(gathered_ids[position], wanted_ids):
        missing = wanted_ids[gathered_ids[position] != wanted_ids][:5]
        raise AssertionError(
            f"replay: no delta came back for vehicles {missing.tolist()}; the "
            "gathered set and the replayed set disagree")

    new_start_index = np.asarray(deltas["new_start_index"])[position].tolist()
    new_offset_m = np.asarray(deltas["new_offset_m"])[position].tolist()
    new_time_offset_us = np.asarray(deltas["new_time_offset_us"])[position].tolist()
    moved_flags = np.asarray(deltas["moved"])[position].astype(bool).tolist()
    changed_segment = np.asarray(deltas["changed_segment"])[position].astype(bool).tolist()
    reached_end = np.asarray(deltas["reached_end"])[position].astype(bool).tolist()

    # The store keeps its own copy of the movement state, and from here it is
    # maintained in production, not just under the shadow check: C++ selection
    # reads these columns, and refreshing them with a Python pass over every
    # vehicle would cost more than the selection saves. Same arrays the loop
    # below uses, so this adds one C++ call per step and no per-vehicle work.
    store_queue_events = store_select.apply_delta_columns(
        wanted_ids, new_start_index, new_offset_m, new_time_offset_us,
        moved_flags, changed_segment, reached_end)

    vehicles_moved = False
    remove_vehicle = queues_manager.remove_vehicle
    add_to_queue = queues_manager.add_to_queue

    for i, vehicle in enumerate(replay_order):
        state = vehicle.__dict__
        old_index = state["start_index"]

        state["start_index"] = new_start_index[i]
        state["start_distance_offset"] = new_offset_m[i]
        state["time_offset"] = timedelta(microseconds=new_time_offset_us[i])

        moved = moved_flags[i]
        vehicles_moved = vehicles_moved or moved

        # Left the previous segment's queue (was at its end, index advanced).
        if changed_segment[i]:
            route = state["osm_route"]
            remove_vehicle(vehicle, route[old_index], route[old_index + 1])

        # Reached the end of the current segment on a moving step: either arrive
        # or queue up. Gating on `moved` keeps blockers and waiters no-ops.
        if reached_end[i] and moved:
            if vehicle.next_node == vehicle.dest_node:
                state["active"] = False
            else:
                add_to_queue(vehicle)

    return vehicles_moved, store_queue_events


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


def _reorder_fcds_to_serial(fcd_arrays: dict, vehicles_to_be_moved: List[Vehicle],
                            free: List[Vehicle], queue_groups: List[List[Vehicle]],
                            moved_by_vid: Dict[int, bool]) -> dict:
    """Reorder the gathered FCDs into serial's exact write order.

    Validation aid (ace.deterministic-order); makes the history byte-identical
    to serial so the HDF5 files can be diffed directly. The workloads and the
    serial path are untouched — this is pure master-side reordering, valid
    because the per-step FCD *set* is identical to serial's.

    route.advance_vehicles_with_queues writes in three passes, and the second
    one is not simply "each queue in FIFO order":

      1. free vehicles, in selection order (route.py:291);
      2. each queue front to back -- but the loop BREAKS at the first vehicle
         that did not move, after writing that vehicle's FCDs (route.py:321).
         Everything behind the blocker is never reached;
      3. every queued vehicle the second pass did not reach, in
         vehicles_in_queues insertion order, i.e. SELECTION order rather than
         queue order (route.py:324).

    An earlier version ranked whole queue groups consecutively, with no notion
    of a blocker. A blocked queue head is the normal case -- that is what a
    queue is -- so everything behind it was ranked mid-step instead of at the
    end, and 54% of the file's rows landed in the wrong place. `moved_by_vid`
    is what makes pass 2 reproducible: the kernel's `moved` flag is exactly
    serial's `prev_pos != vehicle.segment_position` (stalled() sets moved = 0),
    and the deltas are already back by the time this runs.
    """
    order: Dict[int, int] = {}
    k = 0
    for vehicle in free:
        order[vehicle.id] = k; k += 1

    for group in queue_groups:
        blocked = False
        for vehicle in group:
            if blocked:
                continue      # serial never reaches it here; pass 3 ranks it
            order[vehicle.id] = k; k += 1
            # Ranked first, THEN the break tested: serial writes the blocker's
            # own FCDs before stopping.
            blocked = not moved_by_vid.get(vehicle.id, False)

    # Pass 3. Whatever is still unranked is exactly the queued vehicles the
    # second pass did not reach -- those behind a blocker, plus the `waiting`
    # ones that sat behind a selection gap. Walking vehicles_to_be_moved puts
    # them in selection order, which is the order serial's dict iteration gives.
    for vehicle in vehicles_to_be_moved:
        if vehicle.id not in order:
            order[vehicle.id] = k; k += 1

    vids = fcd_arrays["vehicle_id"]
    times = fcd_arrays["datetime_seconds"]
    proc = np.fromiter((order.get(int(v), 1 << 31) for v in vids),
                       dtype=np.int64, count=len(vids))
    perm = np.lexsort((times, proc))   # primary: processing order; secondary: datetime
    return {key: np.asarray(arr)[perm] for key, arr in fcd_arrays.items()}


def advance_vehicles_mpi(vehicles_to_be_moved: List[Vehicle], departure_time,
                         routing_map: Map, queues_manager: QueuesManager,
                         los_vehicles_tolerance, drop_before_time=None,
                         deterministic_order: bool = False,
                         store_shadow_check: bool = False,
                         phase_times: dict = None
                         ) -> Tuple[object, bool]:
    """Distributed equivalent of route.advance_vehicles_with_queues.

    Note: LoS is evaluated against the C++ GlobalView replicas (kept in sync by
    the all-gather), not the Python global_view — so this does not take a gv_db.

    :param deterministic_order: reorder the returned FCDs into serial's write
        order (config.json ace.deterministic-order). Row order only; the records
        themselves and the resulting vehicle positions are identical either way.
    :param store_shadow_check: also build the tasks the old way and require the
        two to be byte-identical (config.json ace.store-shadow-check). Slow.
    :param phase_times: when given, filled with this step's per-phase wall times
        in milliseconds. The step loop merges them into step_timers, so a slow
        advance can be attributed without guessing. Always on -- a few
        perf_counter calls per step, not per vehicle.

    How the tasks are built, how the FCDs come back, how the deltas are
    replayed and how the workloads are packed are NOT options: each was once a
    flag so the new implementation could be measured against the old one, and
    each is now the only implementation. The old task builder (_build_task_array)
    and the old record path (_build_history) survive as the references
    ace.store-shadow-check holds the production path to; nothing else calls them.

    Returns (fcds, vehicles_moved), where fcds is an FCDColumns. Simulation.update
    and FCDHistory.extend both accept it.
    """
    departure_epoch_us = _datetime_to_epoch_us(departure_time)
    los_tolerance_s = los_vehicles_tolerance.total_seconds()

    times = {} if phase_times is None else phase_times
    clock = time.perf_counter
    mark = clock()

    free, queue_groups, waiting = _split_vehicles(vehicles_to_be_moved, queues_manager)
    times["adv_split_ms"] = (clock() - mark) * 1000.0
    mark = clock()

    if store_shadow_check:
        from . import store_bridge
        # Must run right after _split_vehicles and before anything stages a queue
        # operation: split_selected replays that function's defaultdict reads,
        # which is what fixes the registry's key order.
        store_bridge.shadow_check_split(vehicles_to_be_moved, free, queue_groups, waiting)

    flat_queue_vehicles = [v for group in queue_groups for v in group]

    # Build task buffers (structured dtype) + queue CSR offsets.
    from . import store_bridge

    # The store holds last step's movement state until this push; routes and
    # closures were pushed by the caller when they changed. Only the vehicles
    # moving this step need refreshing -- the rest are not asked for a task.
    store_bridge.push_movement(free)
    store_bridge.push_movement(flat_queue_vehicles)
    free_arr = store_bridge.build_tasks(free, ADVANCE_TASK_DTYPE)
    queue_arr = store_bridge.build_tasks(flat_queue_vehicles, ADVANCE_TASK_DTYPE)
    queue_offsets = np.zeros(len(queue_groups) + 1, dtype=np.int32)
    acc = 0
    for i, group in enumerate(queue_groups):
        queue_offsets[i] = acc
        acc += len(group)
    queue_offsets[len(queue_groups)] = acc

    # Shadow mode: the store's tasks and the Python builder's must agree. The
    # store is what shipped; _build_task_array is the reference it is held to,
    # which is the only thing that function is still for. A disagreement is a
    # bug in one of them and stops the run.
    if store_shadow_check:
        from .store_bridge import shadow_check
        # shadow_check builds the store's side itself; this is the other side.
        reference_free = _build_task_array(free, routing_map)
        reference_queue = _build_task_array(flat_queue_vehicles, routing_map)
        shadow_check(free, routing_map, reference_free, ADVANCE_TASK_DTYPE, "free")
        shadow_check(flat_queue_vehicles, routing_map, reference_queue,
                     ADVANCE_TASK_DTYPE, "queues")

    # Hashed here, after the buffers are final and before the kernel reads them:
    # this is exactly what do_advance is handed, whoever built it.
    _emit_task_digest(free_arr, queue_arr)

    times["adv_build_tasks_ms"] = (clock() - mark) * 1000.0
    mark = clock()

    # Prune every GV replica BEFORE do_advance reads it, at the current offset —
    # this reproduces serial's read-time GV (serial prunes at the end of the
    # previous step with the current offset). Doing this in the sync (after the
    # read) pruned one round too late and diverged the LoS. Threshold is in the
    # GV's double-second units, derived from µs to match the FCD datetimes.
    drop_before = (_datetime_to_epoch_us(drop_before_time) / 1e6
                   if drop_before_time is not None else float("-inf"))
    ruthlib.advance_drop_old(drop_before)

    # Phase 1: scatter + evaluate across ranks (reads the just-pruned GV).
    ruthlib.do_advance(free_arr, queue_arr, queue_offsets, departure_epoch_us,
                       los_tolerance_s)

    # After-gap waiting vehicles: inject their stationary FCDs before the sync.
    _inject_waiting_fcds(waiting, routing_map, departure_epoch_us)

    # Phase 2: all-gather the step's FCDs into every GV replica (no pruning here).
    ruthlib.sync_advance()

    # do_advance and sync_advance time themselves internally (dispatch vs
    # barrier vs gather); those come back through advance_phase_times().
    mark = clock()

    deltas = ruthlib.get_advance_deltas()
    fcd_arrays = ruthlib.get_advance_fcds()
    times["adv_readback_ms"] = (clock() - mark) * 1000.0
    mark = clock()

    # Canonical replay order: free vehicles, then each queue group front to
    # back. The queue operations staged here are consumed in this order by
    # batch_update(), so it is part of the semantics, not just tidiness.
    replay_order = list(free)
    for group in queue_groups:
        replay_order.extend(group)

    # Snapshot the staged queue operations so the shadow check can tell which
    # ones this step added, rather than the whole backlog.
    removals_before = len(queues_manager.to_be_removed)
    additions_before = len(queues_manager.to_be_added)

    # Applying deltas to the store must happen EXACTLY ONCE per step. The
    # movement columns are assigned absolutely, so a second pass leaves them
    # unchanged -- but the queue events are derived from the index the vehicle is
    # leaving, which the first pass has already advanced, and advance_waiting ADDS
    # a frequency rather than assigning one. So the second pass reports removals
    # keyed one segment too far and pushes waiting vehicles a step into the
    # future. Whoever applies them keeps the events; nobody re-applies to get
    # their own copy.
    vehicles_moved, store_queue_events = _replay_deltas(
        replay_order, deltas, queues_manager)

    # After-gap waiting: advance time only, no movement, no queue op.
    for vehicle in waiting:
        vehicle.__dict__["time_offset"] = vehicle.time_offset + vehicle.frequency
    if waiting:
        from . import store_select
        store_select.advance_waiting(waiting)

    # Shadow mode: the store replays the same deltas and must land on the same
    # state. Runs BEFORE batch_update, while the staged operations are still
    # comparable; the store does not consume them, it only reports what it would
    # have staged.
    if store_shadow_check:
        from . import store_bridge
        store_bridge.shadow_check_deltas(
            replay_order + list(waiting), store_queue_events,
            queues_manager.to_be_removed[removals_before:],
            queues_manager.to_be_added[additions_before:])
        store_bridge.queues_stage(store_queue_events)

    queues_manager.batch_update()
    times["adv_replay_ms"] = (clock() - mark) * 1000.0
    mark = clock()

    # The registry applies the same staged operations and must reach the same
    # state. Checked after Python's batch_update so both have been applied.
    if store_shadow_check:
        store_bridge.shadow_check_queues(queues_manager)

    # Validation only: reorder gathered FCDs into serial's write order so the FCD
    # history is byte-identical (content is already identical either way).
    if deterministic_order:
        # moved is the kernel's own flag, so this is serial's was_moved exactly.
        moved_by_vid = dict(zip((int(v) for v in deltas["vehicle_id"]),
                                (bool(m) for m in deltas["moved"])))
        fcd_arrays = _reorder_fcds_to_serial(fcd_arrays, vehicles_to_be_moved,
                                             free, queue_groups, moved_by_vid)

    moved_by_id = {v.id: v for v in vehicles_to_be_moved}
    fcds = _build_fcd_columns(fcd_arrays, moved_by_id)
    if store_shadow_check:
        _shadow_check_fcd_columns(fcds, fcd_arrays, moved_by_id, routing_map)

    times["adv_build_fcds_ms"] = (clock() - mark) * 1000.0
    times.update(ruthlib.advance_phase_times())

    return fcds, vehicles_moved


def _shadow_check_fcd_columns(columns: FCDColumns, fcd_arrays: dict,
                              moved_by_id: Dict[int, Vehicle], routing_map: Map):
    """Require the columns to carry exactly what the record path would have.

    The columns feed the global view and the history file, so a wrong `active`
    or a misaligned column changes the output without changing the trajectory --
    which the trajectory digest would not catch.
    """
    records = _build_history(fcd_arrays, moved_by_id, routing_map)
    if len(records) != len(columns):
        raise AssertionError(
            f"fcd columns check: {len(columns)} rows vs {len(records)} records")

    for i, record in enumerate(records):
        expected = (record.vehicle_id,
                    routing_map.segment_id_to_int(record.segment.id),
                    float(record.offset_from_start), float(record.vehicle_speed_mps),
                    bool(record.active))
        got = (int(columns.vehicle_id[i]), int(columns.segment_id[i]),
               float(columns.offset_from_start[i]),
               float(columns.vehicle_speed_mps[i]), bool(columns.active[i]))
        if got != expected:
            names = ("vehicle_id", "segment_id", "offset_from_start",
                     "vehicle_speed_mps", "active")
            detail = ", ".join(f"{name}: columns={g!r} records={e!r}"
                               for name, g, e in zip(names, got, expected) if g != e)
            raise AssertionError(f"fcd columns check: row {i} -> {detail}")
