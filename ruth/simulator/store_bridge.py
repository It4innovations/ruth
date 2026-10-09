"""
Shadow mode for the C++ VehicleStore.

The store is going to take over task construction from `_build_task_array` and
delta replay from `_apply_delta`. Before anything depends on it, this module
runs it *alongside* the Python Vehicle objects and checks, every step, that the
AdvanceTask array the store produces is byte-identical to the one Python
produces. When that holds over a real run, switching the producer over is a
mechanical change rather than a leap.

Enabled by config.json `ace.store-shadow-check`. It is a verification tool, not
a production path: it rebuilds the store from scratch each step, which costs
roughly what `_build_task_array` costs plus a routing-id lookup per route
segment. Use it on short runs, not on the full input.

Rebuilding from scratch rather than incrementally updating is deliberate. An
incremental shadow would need staleness tracking to notice that
`update_followup_route` rewrote a route, and a verifier whose own bookkeeping
can be wrong verifies nothing. Rebuilding cannot go stale.
"""

from datetime import timedelta
from typing import Dict, List, Sequence, Tuple

import numpy as np

import ruthlib

_ONE_US = timedelta(microseconds=1)

# (node_from, node_to) -> routing_id. Map.segment_id_to_int goes through
# networkx's dict-of-dict edge lookup, which is exactly the cost the store
# exists to remove; memoizing keeps the shadow check from dominating the step.
_routing_id_cache: Dict[Tuple[int, int], int] = {}


def _routing_id(routing_map, node_from: int, node_to: int) -> int:
    key = (node_from, node_to)
    rid = _routing_id_cache.get(key)
    if rid is None:
        rid = routing_map.segment_id_to_int(key)
        _routing_id_cache[key] = rid
    return rid


def populate(vehicles: Sequence, routing_map) -> None:
    """Rebuild the store to hold exactly `vehicles`, with their current state."""
    n = len(vehicles)

    vehicle_id = np.empty(n, dtype=np.int32)
    start_index = np.empty(n, dtype=np.int32)
    start_offset_m = np.empty(n, dtype=np.float64)
    time_offset_us = np.empty(n, dtype=np.int64)
    frequency_us = np.empty(n, dtype=np.int64)
    fcd_period_us = np.empty(n, dtype=np.int64)
    active = np.empty(n, dtype=np.int32)
    route_offsets = np.empty(n + 1, dtype=np.int32)

    route_nodes: List[int] = []
    route_segment_ids: List[int] = []

    acc = 0
    for i, vehicle in enumerate(vehicles):
        vehicle_id[i] = vehicle.id
        start_index[i] = vehicle.start_index
        start_offset_m[i] = vehicle.start_distance_offset
        time_offset_us[i] = vehicle.time_offset // _ONE_US
        frequency_us[i] = vehicle.frequency // _ONE_US
        fcd_period_us[i] = vehicle.fcd_sampling_period // _ONE_US
        active[i] = 1 if vehicle.active else 0

        route = vehicle.osm_route
        route_offsets[i] = acc
        route_nodes.extend(route)
        # One segment id per node slot keeps the two CSR buffers aligned; the
        # last node has no outgoing segment, so it gets a -1 that is never read.
        for k in range(len(route) - 1):
            route_segment_ids.append(_routing_id(routing_map, route[k], route[k + 1]))
        route_segment_ids.append(-1)
        acc += len(route)
    route_offsets[n] = acc

    max_routing_id = max(route_segment_ids) if route_segment_ids else 0

    ruthlib.advance_store_reset(n, acc, int(max_routing_id))

    # Only the segments these routes actually touch, deduplicated.
    seen = sorted({rid for rid in route_segment_ids if rid >= 0})
    if seen:
        rid_arr = np.asarray(seen, dtype=np.int64)
        length_m = np.empty(len(seen), dtype=np.float64)
        max_kph = np.empty(len(seen), dtype=np.float64)
        lanes = np.empty(len(seen), dtype=np.int32)
        closed = np.empty(len(seen), dtype=np.int32)
        for j, rid in enumerate(seen):
            node_from, node_to = routing_map.int_to_segment_id(rid)
            segment = routing_map.get_osm_segment(node_from, node_to)
            length_m[j] = segment.length
            max_kph[j] = segment.max_allowed_speed_kph
            lanes[j] = segment.lanes
            closed[j] = 1 if routing_map.get_current_max_speed(node_from, node_to) == 0 else 0
        ruthlib.advance_store_set_segments(rid_arr, length_m, max_kph, lanes, closed)

    ruthlib.advance_store_add_vehicles(
        vehicle_id, start_index, start_offset_m, time_offset_us, frequency_us,
        fcd_period_us, active,
        np.asarray(route_nodes, dtype=np.int64),
        np.asarray(route_segment_ids, dtype=np.int64),
        route_offsets)


def tasks_for(vehicles: Sequence, task_dtype) -> np.ndarray:
    """The AdvanceTask array the store builds for `vehicles`, in that order."""
    ids = np.fromiter((v.id for v in vehicles), dtype=np.int32, count=len(vehicles))
    raw = ruthlib.advance_store_make_tasks(ids)
    return raw.view(task_dtype)


def shadow_check(vehicles: Sequence, routing_map, python_tasks: np.ndarray,
                 task_dtype, label: str) -> None:
    """Compare the store's task array to Python's. Raises on any difference.

    Raises rather than logs: a mismatch means the store and the Python path
    disagree about what the workers should compute, and every later patch in
    this series assumes they agree. Continuing past that would produce results
    whose provenance is unclear, which is worse than stopping.
    """
    if len(vehicles) == 0:
        return

    populate(vehicles, routing_map)
    store_tasks = tasks_for(vehicles, task_dtype)

    if store_tasks.shape != python_tasks.shape:
        raise AssertionError(
            f"store shadow check ({label}): {store_tasks.shape[0]} tasks from the "
            f"store vs {python_tasks.shape[0]} from Python")

    if np.array_equal(store_tasks.view(np.uint8), python_tasks.view(np.uint8)):
        return

    # Name the first offending vehicle and field rather than just "differs".
    for i in range(len(python_tasks)):
        if store_tasks[i].tobytes() == python_tasks[i].tobytes():
            continue
        fields = [name for name in python_tasks.dtype.names
                  if store_tasks[i][name] != python_tasks[i][name]]
        detail = ", ".join(
            f"{name}: store={store_tasks[i][name]!r} python={python_tasks[i][name]!r}"
            for name in fields)
        raise AssertionError(
            f"store shadow check ({label}): row {i}, vehicle "
            f"{python_tasks[i]['vehicle_id']} differs -> {detail}")

    raise AssertionError(
        f"store shadow check ({label}): arrays differ but no row does; "
        "the dtype padding almost certainly disagrees with the C++ struct")


def shadow_check_selection(all_vehicles: Sequence, routing_map,
                           offset_seconds: int, freq_seconds: int,
                           python_selected: Sequence, python_at_segment_end: Sequence,
                           python_current_offset) -> int:
    """Check the store reproduces the step's selection. Returns the count checked.

    Covers the three things singlenode asks the vehicle population each step:
    which vehicles move now (`is_vehicle_within_offset`), which of those sit at a
    segment end (the geometric half of `need_new_route`), and what the next
    offset is (`compute_current_offset`). Both have a detail worth stating,
    because both are reproduced deliberately rather than by accident:

      * the rounding in `is_vehicle_within_offset` is Python's round(), which is
        half-to-EVEN; and
      * `compute_current_offset`'s `filter(None, ...)` drops a time offset of
        exactly zero, not just None.

    Raises on any disagreement, for the reason `shadow_check` does.
    """
    populate(all_vehicles, routing_map)

    store_selected = ruthlib.advance_store_select_within_offset(
        int(offset_seconds), int(freq_seconds))
    expected = np.fromiter((v.id for v in python_selected),
                           dtype=np.int32, count=len(python_selected))
    if not np.array_equal(store_selected, expected):
        raise AssertionError(
            _first_difference("selection", store_selected, expected))

    selected_ids = expected
    store_at_end = ruthlib.advance_store_at_segment_end(selected_ids)
    expected_at_end = np.fromiter((v.id for v in python_at_segment_end),
                                  dtype=np.int32, count=len(python_at_segment_end))
    if not np.array_equal(store_at_end, expected_at_end):
        raise AssertionError(
            _first_difference("at_segment_end", store_at_end, expected_at_end))

    store_offset_us = ruthlib.advance_store_min_active_time_offset_us()
    expected_offset_us = (None if python_current_offset is None
                          else python_current_offset // _ONE_US)
    if store_offset_us != expected_offset_us:
        raise AssertionError(
            f"store shadow check (current_offset): store={store_offset_us} "
            f"python={expected_offset_us} (microseconds)")

    store_active = ruthlib.advance_store_count_active()
    expected_active = sum(1 for v in all_vehicles if v.active)
    if store_active != expected_active:
        raise AssertionError(
            f"store shadow check (active count): store={store_active} "
            f"python={expected_active}")

    return len(all_vehicles)


def _first_difference(label: str, store_ids: np.ndarray, python_ids: np.ndarray) -> str:
    """Name the first differing id rather than just reporting inequality."""
    if len(store_ids) != len(python_ids):
        extra = sorted(set(store_ids.tolist()) ^ set(python_ids.tolist()))[:10]
        return (f"store shadow check ({label}): {len(store_ids)} ids from the store "
                f"vs {len(python_ids)} from Python; symmetric difference starts "
                f"{extra}")
    for i, (a, b) in enumerate(zip(store_ids.tolist(), python_ids.tolist())):
        if a != b:
            return (f"store shadow check ({label}): position {i} is vehicle {a} in "
                    f"the store, {b} in Python")
    return f"store shadow check ({label}): arrays differ but no position does"
