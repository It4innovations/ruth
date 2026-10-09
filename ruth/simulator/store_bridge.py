"""
The C++ VehicleStore as seen from Python: population, upkeep, and verification.

The store is now the PRODUCER of the per-step AdvanceTask array (config.json
`ace.store-tasks`, on by default). That is the point of the whole exercise:
`_build_task_array` cost about 8.4 us per vehicle, most of it networkx
dict-of-dict lookups (`segment_id_to_int`, `get_current_max_speed`) and eighteen
boxed field stores per row. Against a persistent store that work disappears --
the per-step Python cost becomes four column pushes per moving vehicle, and the
task itself is built in C++ from data already resident on every rank.

Keeping it correct means keeping it in step with the Python objects, which are
still authoritative until the store takes over delta replay. Three things can
change under it, and each has an explicit push:

  * movement state, every step          -> push_movement()
  * a route, when route selection rewrites one -> push_routes()
  * a segment closure, when the map changes    -> push_closures()

and the population itself (streaming loads, pruning) is watched by
ensure_population(), which re-initialises rather than trying to patch up the
difference.

`_build_task_array` is NOT deleted. It stays as the reference implementation the
shadow check compares against (`ace.store-shadow-check`), and as the fallback
when `ace.store-tasks` is off, which is how you A/B the two. A producer with no
reference to check against is the thing this series has been avoiding.
"""

from datetime import timedelta
from typing import Dict, List, Sequence, Tuple

import numpy as np

import ruthlib

_ONE_US = timedelta(microseconds=1)

def _routing_id(routing_map, node_from: int, node_to: int) -> int:
    """(node_from, node_to) -> routing_id, memoised on the map.

    Map.segment_id_to_int goes through networkx's dict-of-dict edge lookup,
    which is exactly the cost the store exists to remove. The cache hangs off
    the map rather than this module because routing ids belong to the map that
    assigned them: a module-global would survive a map swap and hand back ids
    for the wrong graph.
    """
    cache = getattr(routing_map, "_store_routing_ids", None)
    if cache is None:
        cache = {}
        routing_map._store_routing_ids = cache

    key = (node_from, node_to)
    rid = cache.get(key)
    if rid is None:
        rid = routing_map.segment_id_to_int(key)
        cache[key] = rid
    return rid


# Vehicle ids the store was last initialised with, as an array, so
# ensure_population() can spot a changed population with one vectorised compare
# instead of trusting the caller to tell us.
_population_ids: "np.ndarray | None" = None


def initialize(vehicles: Sequence, routing_map) -> None:
    """Rebuild the store to hold exactly `vehicles`, with their current state."""
    global _population_ids
    n = len(vehicles)

    vehicle_id = np.empty(n, dtype=np.int32)
    start_index = np.empty(n, dtype=np.int32)
    start_offset_m = np.empty(n, dtype=np.float64)
    time_offset_us = np.empty(n, dtype=np.int64)
    frequency_us = np.empty(n, dtype=np.int64)
    fcd_period_us = np.empty(n, dtype=np.int64)
    active = np.empty(n, dtype=np.int32)
    dest_node = np.empty(n, dtype=np.int64)
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
        dest_node[i] = vehicle.dest_node

        route = vehicle.osm_route
        route_offsets[i] = acc
        route_nodes.extend(route)
        _append_segment_ids(route_segment_ids, route, routing_map)
        acc += len(route)
    route_offsets[n] = acc

    _assert_csr_aligned(route_nodes, route_segment_ids, acc)

    max_routing_id = max(route_segment_ids) if route_segment_ids else 0

    ruthlib.advance_store_reset(n, acc, int(max_routing_id))

    _push_segment_geometry(routing_map, route_segment_ids)

    ruthlib.advance_store_add_vehicles(
        vehicle_id, start_index, start_offset_m, time_offset_us, frequency_us,
        fcd_period_us, active, dest_node,
        np.asarray(route_nodes, dtype=np.int64),
        np.asarray(route_segment_ids, dtype=np.int64),
        route_offsets)

    _population_ids = vehicle_id.copy()


def ensure_population(vehicles: Sequence, routing_map) -> bool:
    """Re-initialise the store if the vehicle population changed. True if it did.

    Streaming loads vehicles mid-run and pruning drops them, so the store cannot
    assume the list it was built from still holds. Comparing the id vectors is
    O(N) of vectorised work per step, which is cheap next to the per-vehicle
    Python the store exists to remove, and it cannot get out of step the way a
    caller-maintained dirty flag could.
    """
    ids = np.fromiter((v.id for v in vehicles), dtype=np.int32, count=len(vehicles))
    if _population_ids is not None and np.array_equal(ids, _population_ids):
        return False
    initialize(vehicles, routing_map)
    return True


def push_movement(vehicles: Sequence) -> int:
    """Refresh the store's movement columns for `vehicles`. Returns the count.

    This is the per-step cost of the persistent store: four attribute reads and
    four array stores per vehicle, and no map lookups at all.
    """
    n = len(vehicles)
    if n == 0:
        return 0

    vehicle_id = np.empty(n, dtype=np.int32)
    start_index = np.empty(n, dtype=np.int32)
    start_offset_m = np.empty(n, dtype=np.float64)
    time_offset_us = np.empty(n, dtype=np.int64)
    active = np.empty(n, dtype=np.int32)

    for i, vehicle in enumerate(vehicles):
        vehicle_id[i] = vehicle.id
        start_index[i] = vehicle.start_index
        start_offset_m[i] = vehicle.start_distance_offset
        time_offset_us[i] = vehicle.time_offset // _ONE_US
        active[i] = 1 if vehicle.active else 0

    ruthlib.advance_store_update_movement(
        vehicle_id, start_index, start_offset_m, time_offset_us, active)
    return n


def push_routes(vehicles: Sequence, routing_map) -> int:
    """Replace the store's routes for `vehicles`. Returns the count.

    Called for the vehicles whose route selection rewrote this step -- a few
    percent of the active set, not all of it.
    """
    n = len(vehicles)
    if n == 0:
        return 0

    vehicle_id = np.empty(n, dtype=np.int32)
    start_index = np.empty(n, dtype=np.int32)
    route_offsets = np.empty(n + 1, dtype=np.int32)
    route_nodes: List[int] = []
    route_segment_ids: List[int] = []

    acc = 0
    for i, vehicle in enumerate(vehicles):
        vehicle_id[i] = vehicle.id
        start_index[i] = vehicle.start_index
        route = vehicle.osm_route
        route_offsets[i] = acc
        route_nodes.extend(route)
        _append_segment_ids(route_segment_ids, route, routing_map)
        acc += len(route)
    route_offsets[n] = acc

    _assert_csr_aligned(route_nodes, route_segment_ids, acc)

    # A rewritten route may run over segments no route touched before, so their
    # geometry has to reach the table before a task can be built from it.
    _push_segment_geometry(routing_map, route_segment_ids)

    ruthlib.advance_store_set_routes(
        vehicle_id, start_index,
        np.asarray(route_nodes, dtype=np.int64),
        np.asarray(route_segment_ids, dtype=np.int64),
        route_offsets)
    return n


def push_closures(segment_keys, routing_map) -> int:
    """Update the store's closure flags for the given (node_from, node_to) keys.

    `is_closed` mirrors `get_current_max_speed(...) == 0`, and the only thing
    that writes that field is Map.update_temporary_max_speeds -- which returns
    exactly the segments it touched, so this is fed from its return value rather
    than by rescanning the network.
    """
    keys = list(segment_keys)
    if not keys:
        return 0

    routing_id = np.empty(len(keys), dtype=np.int64)
    is_closed = np.empty(len(keys), dtype=np.int32)
    for i, (node_from, node_to) in enumerate(keys):
        routing_id[i] = _routing_id(routing_map, node_from, node_to)
        is_closed[i] = 1 if routing_map.get_current_max_speed(node_from, node_to) == 0 else 0

    ruthlib.advance_store_set_closures(routing_id, is_closed)
    return len(keys)


def build_tasks(vehicles: Sequence, task_dtype) -> np.ndarray:
    """The production AdvanceTask array, built in C++ from the store."""
    ids = np.fromiter((v.id for v in vehicles), dtype=np.int32, count=len(vehicles))
    raw = ruthlib.advance_store_build_tasks(ids)
    return raw.view(task_dtype)


def _append_segment_ids(route_segment_ids: List[int], route, routing_map) -> None:
    """Append one segment id per node slot of `route`, keeping the CSR aligned.

    A route of N nodes contributes exactly N entries: its N-1 segment ids plus
    one padding slot for the destination node, which has no outgoing segment and
    is never read.

    An EMPTY route contributes NOTHING. That is the whole point of the guard:
    vehicles with osm_route == [] are real (change_baseline_alternatives empties
    the route of any vehicle it cannot route, and they stay in sim.vehicles
    because pruning only happens when streaming), and padding them pushed an
    entry here while the node offset stood still. Every vehicle after one of
    them then read another vehicle's segment ids -- alongside its own correct
    nodes, which is exactly what made the symptom look like a stale route.
    """
    if not route:
        return
    for k in range(len(route) - 1):
        route_segment_ids.append(_routing_id(routing_map, route[k], route[k + 1]))
    route_segment_ids.append(-1)


def _assert_csr_aligned(route_nodes: List[int], route_segment_ids: List[int],
                        expected: int) -> None:
    """Both CSR buffers must be indexed by the same offsets."""
    if len(route_nodes) != expected or len(route_segment_ids) != expected:
        raise AssertionError(
            f"store CSR drifted: {len(route_nodes)} nodes, "
            f"{len(route_segment_ids)} segment ids, offsets expect {expected}")


def _push_segment_geometry(routing_map, route_segment_ids) -> None:
    """Fill the segment table for every routing id in `route_segment_ids`."""
    seen = sorted({int(rid) for rid in route_segment_ids if rid >= 0})
    if not seen:
        return

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

    # No repopulation: the store is live and the caller has already pushed this
    # step's movement into it. Rebuilding here would verify a store that the
    # simulation is not the one using, which is worse than not checking.
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
        vehicle_id = int(python_tasks[i]["vehicle_id"])
        raise AssertionError(
            f"store shadow check ({label}): row {i}, vehicle {vehicle_id} "
            f"differs -> {detail}\n"
            + describe_vehicle(vehicle_id, vehicles[i], routing_map))

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

    Raises on any disagreement, for the reason `shadow_check` does. The caller
    keeps the store current; this only reads it.
    """
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


def describe_vehicle(vehicle_id: int, vehicle, routing_map) -> str:
    """Side-by-side of what the store holds for a vehicle and what Python holds.

    A task mismatch on the segment fields is nearly always one of two things:
    the store's route is not the current one (a push was missed), or its
    start_index is. Both show up immediately here, which beats reasoning about
    which push should have fired.
    """
    info = ruthlib.advance_store_debug_vehicle(int(vehicle_id))
    if not info.get("found"):
        return f"    store: vehicle {vehicle_id} is NOT in the store"

    store_nodes = [int(x) for x in info["route_nodes"]]
    python_nodes = [int(x) for x in vehicle.osm_route]
    store_index = int(info["start_index"])
    python_index = int(vehicle.start_index)

    lines = [
        f"    start_index   store={store_index} python={python_index}",
        f"    start_offset  store={float(info['start_offset_m'])!r} "
        f"python={float(vehicle.start_distance_offset)!r}",
        f"    route_len     store={len(store_nodes)} python={len(python_nodes)}",
        f"    dest_node     store={int(info['dest_node'])} python={vehicle.dest_node}",
    ]

    if store_nodes != python_nodes:
        first = next((k for k, (a, b) in enumerate(zip(store_nodes, python_nodes))
                      if a != b), min(len(store_nodes), len(python_nodes)))
        lines.append(f"    ROUTES DIFFER from node {first}: "
                     f"store={store_nodes[first:first + 4]} "
                     f"python={python_nodes[first:first + 4]}")
        lines.append("    -> the store's route is stale; a push_routes was missed "
                     "for this vehicle")
    else:
        lines.append("    routes are identical; the mismatch is in the index or "
                     "the segment table")
        window = python_nodes[python_index:python_index + 3]
        lines.append(f"    python window at index {python_index}: {window}")
        lines.append(f"    their routing ids: "
                     f"{[_routing_id(routing_map, a, b) for a, b in zip(window, window[1:])]}")
        lines.append(f"    store segment ids at that index: "
                     f"{[int(x) for x in info['route_segment_ids'][store_index:store_index + 2]]}")
    return "\n".join(lines)


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


def apply_deltas(ordered_vehicles: Sequence, delta_by_id) -> dict:
    """Apply this step's deltas to the store, in `ordered_vehicles` order.

    The order is not incidental. Python replays free vehicles first, then each
    queue group front to back, and the queue operations it stages are consumed
    in that order by QueuesManager.batch_update(); the store has to see the same
    sequence or the queues end up holding the same ids in a different order.
    """
    n = len(ordered_vehicles)
    vehicle_id = np.empty(n, dtype=np.int32)
    new_start_index = np.empty(n, dtype=np.int32)
    new_offset_m = np.empty(n, dtype=np.float64)
    new_time_offset_us = np.empty(n, dtype=np.int64)
    moved = np.empty(n, dtype=np.int32)
    changed_segment = np.empty(n, dtype=np.int32)
    reached_end = np.empty(n, dtype=np.int32)

    for i, vehicle in enumerate(ordered_vehicles):
        d = delta_by_id[vehicle.id]
        vehicle_id[i] = vehicle.id
        new_start_index[i] = int(d["new_start_index"])
        new_offset_m[i] = float(d["new_offset_m"])
        new_time_offset_us[i] = int(d["new_time_offset_us"])
        moved[i] = 1 if d["moved"] else 0
        changed_segment[i] = 1 if d["changed_segment"] else 0
        reached_end[i] = 1 if d["reached_end"] else 0

    return ruthlib.advance_store_apply_deltas(
        vehicle_id, new_start_index, new_offset_m, new_time_offset_us,
        moved, changed_segment, reached_end)


def advance_waiting(vehicles: Sequence) -> None:
    """Advance the store's clock for vehicles that waited this step."""
    if not vehicles:
        return
    ids = np.fromiter((v.id for v in vehicles), dtype=np.int32, count=len(vehicles))
    ruthlib.advance_store_advance_waiting(ids)


def shadow_check_deltas(vehicles: Sequence, queue_events: dict,
                        python_removals, python_additions) -> None:
    """Check the store's post-replay state matches the Python objects'.

    Two halves, because delta application has two effects. The movement columns
    must match the Vehicle attributes, and the queue operations the store staged
    must match the ones QueuesManager staged -- same operations, same order,
    since batch_update() pops queues front to back and would otherwise assert.
    """
    columns = ruthlib.advance_store_movement_columns()
    by_id = {int(v): i for i, v in enumerate(columns["vehicle_id"])}

    for vehicle in vehicles:
        i = by_id.get(vehicle.id)
        if i is None:
            raise AssertionError(
                f"store shadow check (deltas): vehicle {vehicle.id} is not in the store")

        expected = (vehicle.start_index, vehicle.start_distance_offset,
                    vehicle.time_offset // _ONE_US, 1 if vehicle.active else 0)
        got = (int(columns["start_index"][i]), float(columns["start_offset_m"][i]),
               int(columns["time_offset_us"][i]), int(columns["active"][i]))
        if got != expected:
            names = ("start_index", "start_offset_m", "time_offset_us", "active")
            detail = ", ".join(f"{name}: store={g!r} python={e!r}"
                               for name, g, e in zip(names, got, expected) if g != e)
            raise AssertionError(
                f"store shadow check (deltas): vehicle {vehicle.id} -> {detail}")

    _compare_queue_events("removals", queue_events["removals"], python_removals)
    _compare_queue_events("additions", queue_events["additions"], python_additions)


def _compare_queue_events(kind: str, store_events: dict, python_events) -> None:
    """Compare one kind of staged queue operation, in order."""
    store_rows = list(zip((int(x) for x in store_events["vehicle_id"]),
                          (int(x) for x in store_events["node_from"]),
                          (int(x) for x in store_events["node_to"])))
    python_rows = [(int(vid), int(node_from), int(node_to))
                   for (vid, node_from, node_to) in python_events]

    if store_rows == python_rows:
        return

    if len(store_rows) != len(python_rows):
        raise AssertionError(
            f"store shadow check (queue {kind}): store staged {len(store_rows)} "
            f"operations, Python staged {len(python_rows)}")
    for i, (got, expected) in enumerate(zip(store_rows, python_rows)):
        if got != expected:
            raise AssertionError(
                f"store shadow check (queue {kind}): operation {i} is "
                f"{got} in the store, {expected} in Python")


def queues_stage(queue_events: dict) -> None:
    """Stage the queue operations apply_deltas reported, removals then additions."""
    removals, additions = queue_events["removals"], queue_events["additions"]
    ruthlib.advance_queues_stage(
        removals["vehicle_id"], removals["node_from"], removals["node_to"],
        additions["vehicle_id"], additions["node_from"], additions["node_to"])


def shadow_check_queues(queues_manager) -> int:
    """Apply the staged ops in C++ and require the result to match Python's.

    Compares the whole structure, not just membership: the keys in creation
    order, and the ids in each queue front to back. Both matter -- key order
    decides how queue groups are built and dispatched, and position within a
    queue decides which vehicle blocks which.

    Returns the number of queues compared.
    """
    ruthlib.advance_queues_batch_update()
    snap = ruthlib.advance_queues_snapshot()

    store_keys = [(int(a), int(b))
                  for a, b in zip(snap["node_from"], snap["node_to"])]
    python_keys = [(int(a), int(b)) for (a, b) in queues_manager.queues.keys()]

    if store_keys != python_keys:
        for i, (got, expected) in enumerate(zip(store_keys, python_keys)):
            if got != expected:
                raise AssertionError(
                    f"store shadow check (queue keys): position {i} is {got} in the "
                    f"registry, {expected} in Python "
                    f"({len(store_keys)} keys vs {len(python_keys)})")
        raise AssertionError(
            f"store shadow check (queue keys): {len(store_keys)} keys in the registry, "
            f"{len(python_keys)} in Python; the shorter is a prefix of the longer")

    offsets, ids = snap["offsets"], snap["vehicle_id"]
    for i, key in enumerate(store_keys):
        store_queue = [int(x) for x in ids[offsets[i]:offsets[i + 1]]]
        python_queue = [int(x) for x in queues_manager.queues[key]]
        if store_queue != python_queue:
            raise AssertionError(
                f"store shadow check (queue {key}): registry has {store_queue}, "
                f"Python has {python_queue}")
    return len(store_keys)


def shadow_check_split(selected: Sequence, python_free: Sequence,
                       python_queue_groups: Sequence, python_waiting: Sequence) -> int:
    """Check the C++ partition matches _split_vehicles'. Returns vehicles split.

    Compares all four observable orderings, not just membership: `free` order,
    the number and order of queue groups, each group's front-to-back contents,
    and `waiting` order. Group order decides dispatch order, position within a
    group decides which vehicle blocks which, and `waiting` order feeds the
    stationary FCDs -- so a set comparison would pass on states that later
    diverge.

    Pushes movement first: the partition is read from each vehicle's current
    position in its route, so a store still holding last step's start_index
    would partition a different world.
    """
    push_movement(selected)

    ids = np.fromiter((v.id for v in selected), dtype=np.int32, count=len(selected))
    result = ruthlib.advance_queues_split(ids)

    store_free = [int(x) for x in result["free"]]
    expected_free = [int(v.id) for v in python_free]
    if store_free != expected_free:
        raise AssertionError(_first_difference("split free",
                                               np.asarray(store_free, dtype=np.int32),
                                               np.asarray(expected_free, dtype=np.int32)))

    offsets = [int(x) for x in result["queue_offsets"]]
    queue_ids = [int(x) for x in result["queue_ids"]]
    store_groups = [queue_ids[offsets[g]:offsets[g + 1]] for g in range(len(offsets) - 1)]
    expected_groups = [[int(v.id) for v in group] for group in python_queue_groups]
    if store_groups != expected_groups:
        if len(store_groups) != len(expected_groups):
            raise AssertionError(
                f"store shadow check (split groups): {len(store_groups)} groups in the "
                f"registry, {len(expected_groups)} in Python")
        for g, (got, expected) in enumerate(zip(store_groups, expected_groups)):
            if got != expected:
                raise AssertionError(
                    f"store shadow check (split group {g}): registry has {got}, "
                    f"Python has {expected}")

    store_waiting = [int(x) for x in result["waiting"]]
    expected_waiting = [int(v.id) for v in python_waiting]
    if store_waiting != expected_waiting:
        raise AssertionError(_first_difference("split waiting",
                                               np.asarray(store_waiting, dtype=np.int32),
                                               np.asarray(expected_waiting, dtype=np.int32)))
    return len(selected)
