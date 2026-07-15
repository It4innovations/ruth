import hashlib
import logging
import math
from datetime import datetime, timedelta
from typing import List, Tuple

from .queues import QueuesManager
from ..data.map import Map
from ..data.segment import Segment, SegmentPosition, SpeedMps, SpeedKph, LengthMeters
from .simulation import FCDRecord
from ..globalview_heterogeneous import GlobalView

logger = logging.getLogger(__name__)


def _get_speed_limit_kph(highway_type: str, vehicle_type: str) -> float:
    speed_limits = {
        "motorway": {"car": 130, "truck": 80},
        "motorway_link": {"car": 100, "truck": 60},
        "trunk": {"car": 100, "truck": 70},
        "trunk_link": {"car": 80, "truck": 50},
        "primary": {"car": 90, "truck": 60},
        "primary_link": {"car": 70, "truck": 50},
        "secondary": {"car": 80, "truck": 50},
        "secondary_link": {"car": 60, "truck": 40},
        "tertiary": {"car": 70, "truck": 40},
        "tertiary_link": {"car": 50, "truck": 30},
        "unclassified": {"car": 50, "truck": 30},
        "residential": {"car": 50, "truck": 30},
        "living_street": {"car": 20, "truck": 20},
        "service": {"car": 30, "truck": 30},
        "track": {"car": 30, "truck": 20},
        "footway": {"car": 10, "truck": 10},
        "cycleway": {"car": 10, "truck": 10},
        "path": {"car": 10, "truck": 10},
        "pedestrian": {"car": 10, "truck": 10},
        "steps": {"car": 5, "truck": 5},
    }
    return speed_limits.get(highway_type, {}).get(vehicle_type, 50)


def _norm_vtype(vt) -> str:
    if isinstance(vt, (bytes, bytearray)):
        return vt.decode("utf-8", errors="ignore")
    return str(vt)


def _as_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float(str(x))


def _stable_driver_factor(vehicle_id: int) -> float:
    """
    Stable per-vehicle factor in [0.92, 1.08].
    Deterministic: same vehicle_id -> same factor forever.
    """
    h = hashlib.md5(str(int(vehicle_id)).encode()).hexdigest()
    x = int(h[:8], 16)
    return 0.92 + (x % 1600) / 10000.0  # 0.92 .. 1.08


def _edge_highway_type(routing_map: Map, seg: Segment) -> str:
    u, v = seg.node_from, seg.node_to
    try:
        data = routing_map.current_network[u][v][0]
        hw = data.get("highway", "unclassified")
        if isinstance(hw, list) and hw:
            return str(hw[0])
        return str(hw)
    except Exception:
        return "unclassified"


def _edge_freeflow_kph(routing_map: Map, seg: Segment) -> float:
    """
    Best-effort free-flow speed (kph) from edge data.
    Priority: current_speed -> speed_kph -> maxspeed -> seg.max_allowed_speed_kph
    """
    u, v = seg.node_from, seg.node_to

    try:
        data = routing_map.current_network[u][v][0]

        if data.get("current_speed") is not None:
            k = _as_float(data["current_speed"])
            if k > 0:
                return k

        if data.get("speed_kph") is not None:
            k = _as_float(data["speed_kph"])
            if k > 0:
                return k

        if data.get("maxspeed") is not None:
            ms = data["maxspeed"]
            if isinstance(ms, list) and ms:
                k = _as_float(ms[0])
            else:
                k = _as_float(ms)
            if k > 0:
                return k
    except Exception:
        pass

    try:
        k = _as_float(getattr(seg, "max_allowed_speed_kph", 0.0))
        if k > 0:
            return k
    except Exception:
        pass

    return 0.0


def _compute_turn_type(routing_map: Map, prev_seg: Segment, next_seg: Segment) -> Tuple[str, float]:
    """
    Returns (turn_type, abs_angle_deg)
      turn_type: straight | turn | u_turn
    """
    try:
        if prev_seg.node_from == next_seg.node_from and prev_seg.node_to == next_seg.node_to:
            return "straight", 0.0

        n = routing_map.network.nodes
        from_pos = (n[prev_seg.node_from]["x"], n[prev_seg.node_from]["y"])
        to_pos = (n[prev_seg.node_to]["x"], n[prev_seg.node_to]["y"])
        next_to_pos = (n[next_seg.node_to]["x"], n[next_seg.node_to]["y"])

        cur_vec = (to_pos[0] - from_pos[0], to_pos[1] - from_pos[1])
        nxt_vec = (next_to_pos[0] - to_pos[0], next_to_pos[1] - to_pos[1])

        cur_ang = math.atan2(cur_vec[1], cur_vec[0])
        nxt_ang = math.atan2(nxt_vec[1], nxt_vec[0])
        diff = nxt_ang - cur_ang

        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi

        abs_ang = abs(diff)
        abs_deg = abs_ang * 180.0 / math.pi

        if abs_ang < math.radians(30):
            return "straight", abs_deg
        if abs_ang > math.radians(150):
            return "u_turn", abs_deg
        return "turn", abs_deg
    except Exception:
        return "straight", 0.0


def move_on_segment(
    vehicle,
    driving_route_part: List[Segment],
    current_time: datetime,
    gv_db: GlobalView,
    routing_map: Map,
    los_vehicles_tolerance: timedelta = timedelta(seconds=0),
) -> Tuple[datetime, SegmentPosition, SpeedMps]:
    """
    Move vehicle for one tick (vehicle.frequency).
    Returns (end_time, new_segment_position, effective_speed_mps_over_tick).
    """

    tick_s = vehicle.frequency.total_seconds()
    end_time = current_time + vehicle.frequency

    if not driving_route_part or tick_s <= 0:
        return end_time, vehicle.segment_position, SpeedMps(0.0)

    vtype = _norm_vtype(getattr(vehicle, "vehicle_type", "car"))

    seg_index = int(vehicle.segment_position.index)
    pos_m = _as_float(vehicle.segment_position.position)

    # The "current segment" in this tick is the segment at driving_route_part[0]
    current_seg = driving_route_part[0]
    seg_len_m = _as_float(current_seg.length)
    if pos_m > seg_len_m:
        pos_m = seg_len_m

    # -------------------------
    # IMPORTANT FIX: if already at end, jump to next segment start (index+1)
    # -------------------------
    if math.isclose(pos_m, seg_len_m, abs_tol=1e-6):
        # last segment -> nothing to do
        if seg_index >= (len(vehicle.osm_route) - 1):
            return end_time, vehicle.segment_position, SpeedMps(0.0)

        # next segment closed -> wait
        if vehicle.has_next_segment_closed(routing_map):
            return end_time, vehicle.segment_position, SpeedMps(0.0)

        # move to next segment at 0
        seg_index += 1
        pos_m = 0.0

        # if we have the next segment in driving_route_part, use it
        if len(driving_route_part) > 1:
            current_seg = driving_route_part[1]
            seg_len_m = _as_float(current_seg.length)
        else:
            # fallback: build a new segment from osm nodes
            u = vehicle.osm_route[seg_index]
            v = vehicle.osm_route[seg_index + 1]
            current_seg = routing_map.osm_route_to_py_segments([u, v])[0]
            seg_len_m = _as_float(current_seg.length)

    # -------------------------
    # Turn penalty (only matters if we started exactly at boundary and just entered new segment)
    # We compute it using prev (old) and new segments if available.
    # -------------------------
    turn_penalty_s = 0.0
    if pos_m <= 1e-9 and len(driving_route_part) > 1:
        # We entered a new segment this tick (or are at start), and we can look at prev->next
        prev_seg = driving_route_part[0]
        next_seg = driving_route_part[1]
        turn_type, abs_deg = _compute_turn_type(routing_map, prev_seg, next_seg)
        if turn_type != "straight":
            if vtype == "truck":
                turn_penalty_s = 1.8 if turn_type == "turn" else 3.2
            else:
                turn_penalty_s = 1.0 if turn_type == "turn" else 2.0

            # If traffic is already heavy, turning costs more time
            # (keeps turning effects visible)
            # We'll apply this after LoS is computed.

    if turn_penalty_s >= tick_s:
        return end_time, SegmentPosition(seg_index, LengthMeters(pos_m)), SpeedMps(0.0)

    # -------------------------
    # Level of Service
    # -------------------------
    los = gv_db.level_of_service_in_front_of_vehicle(
        current_time,
        current_seg,
        vehicle.id,
        LengthMeters(pos_m),
        los_vehicles_tolerance,
        limit_vehicle_count=(pos_m <= 1e-9),
    )

    if los == float("inf"):
        return end_time, SegmentPosition(seg_index, LengthMeters(pos_m)), SpeedMps(0.0)

    los = float(los)

    # Increase turn penalty slightly in congestion (optional but realistic)
    if turn_penalty_s > 0 and los < 0.6:
        turn_penalty_s *= 1.25
        if turn_penalty_s >= tick_s:
            return end_time, SegmentPosition(seg_index, LengthMeters(pos_m)), SpeedMps(0.0)

    # -------------------------
    # Free-flow speed on this edge (kph)
    # -------------------------
    ff_kph = _edge_freeflow_kph(routing_map, current_seg)
    if ff_kph <= 0.0:
        return end_time, SegmentPosition(seg_index, LengthMeters(pos_m)), SpeedMps(0.0)

    # Cap by vehicle-type legal/desired speed limit (truck slower than car!)
    highway_type = _edge_highway_type(routing_map, current_seg)
    type_limit_kph = float(_get_speed_limit_kph(highway_type, vtype))
    if type_limit_kph > 0:
        ff_kph = min(ff_kph, type_limit_kph)

    # -------------------------
    # Convert LoS to traffic factor (nonlinear to avoid flat 2-value speeds)
    #  - los=1.0 => ~1.0
    #  - los small => drops smoothly
    # -------------------------
    traffic_factor = 0.20 + 0.80 * (los ** 1.6)

    # Base speed in m/s
    speed_mps = (ff_kph / 3.6) * traffic_factor

    # Stable per-vehicle driver factor -> breaks ties without randomness
    speed_mps *= _stable_driver_factor(vehicle.id)

    # Clamp by vehicle maximum speed (your VehicleClassParams max_speed_mps)
    vmax = _as_float(getattr(vehicle, "max_speed_mps", speed_mps))
    speed_mps = min(speed_mps, vmax)

    if speed_mps <= 1e-9:
        return end_time, SegmentPosition(seg_index, LengthMeters(pos_m)), SpeedMps(0.0)

    # -------------------------
    # Move distance inside tick (after turn penalty time)
    # -------------------------
    move_time_s = max(tick_s - turn_penalty_s, 0.0)

    remaining_m = max(seg_len_m - pos_m, 0.0)
    possible_move_m = speed_mps * move_time_s

    if possible_move_m >= remaining_m and speed_mps > 1e-9:
        travel_time_s = turn_penalty_s + (remaining_m / speed_mps)
        new_pos_m = seg_len_m
        end_time = current_time + timedelta(seconds=travel_time_s)
        effective_speed_mps = speed_mps
    else:
        new_pos_m = pos_m + possible_move_m
        end_time = current_time + vehicle.frequency
        effective_speed_mps = speed_mps

    return end_time, SegmentPosition(seg_index, LengthMeters(new_pos_m)), SpeedMps(effective_speed_mps)


def advance_vehicle(
    vehicle,
    departure_time: datetime,
    gv_db: GlobalView,
    routing_map: Map,
    queues_manager: QueuesManager,
    los_vehicles_tolerance: timedelta = timedelta(seconds=0),
) -> List[FCDRecord]:
    """Advance a vehicle on a route."""
    current_time = departure_time + vehicle.time_offset

    old_pos = vehicle.segment_position

    # Build a short lookahead route (old index .. old index+2)
    idx = int(vehicle.segment_position.index)
    osm_route_part = vehicle.osm_route[idx: idx + 3]
    driving_route_part = routing_map.osm_route_to_py_segments(osm_route_part)

    vehicle_end_time, new_pos, assigned_speed_mps = move_on_segment(
        vehicle, driving_route_part, current_time, gv_db, routing_map, los_vehicles_tolerance
    )

    # If vehicle couldn't move at all on its first segment at time start, just advance time
    if new_pos.index == 0 and math.isclose(_as_float(new_pos.position), 0.0, abs_tol=1e-9):
        vehicle.time_offset += vehicle.frequency
        return []

    # Update vehicle state
    vehicle.time_offset += vehicle_end_time - current_time
    vehicle.set_position(new_pos)

    # Determine segments for queue operations
    # Segment at old position:
    seg_old = driving_route_part[0]

    # Segment at new position (may be next in driving_route_part)
    if new_pos.index != old_pos.index and len(driving_route_part) > 1:
        seg_new = driving_route_part[1]
    else:
        seg_new = driving_route_part[0]

    # Remove from outdated queue if it changed segment
    if (_as_float(old_pos.position) == _as_float(seg_old.length)
            and old_pos.index != vehicle.segment_position.index):
        node_from, node_to = vehicle.osm_route[old_pos.index], vehicle.osm_route[old_pos.index + 1]
        queues_manager.remove_vehicle(vehicle, node_from, node_to)

    # If at end of current segment, handle queue/destination
    if math.isclose(_as_float(vehicle.segment_position.position), _as_float(seg_new.length), abs_tol=1e-6):
        if vehicle.next_node == vehicle.dest_node:
            vehicle.active = False
            queues_manager.remove_inactive_vehicle(vehicle)
        elif old_pos != vehicle.segment_position:
            queues_manager.add_to_queue(vehicle)

    # Generate FCD if still within route bounds
    if vehicle.segment_position.index < (len(vehicle.osm_route) - 1):
        return generate_fcds(
            current_time,
            vehicle_end_time,
            old_pos,
            vehicle.segment_position,
            assigned_speed_mps,
            vehicle,
            [seg_new],  # generate on the current segment only
            remains_active=vehicle.active,
        )

    return []


def advance_waiting_vehicle(vehicle, routing_map: Map, departure_time: datetime) -> List[FCDRecord]:
    current_time = departure_time + vehicle.time_offset

    idx = int(vehicle.start_index)
    osm_route_part = vehicle.osm_route[idx: idx + 2]
    driving_route_part = routing_map.osm_route_to_py_segments(osm_route_part)

    vehicle_end_time = current_time + vehicle.frequency
    assigned_speed_mps = SpeedMps(0.0)

    vehicle.time_offset += vehicle_end_time - current_time

    return generate_fcds(
        current_time,
        vehicle_end_time,
        vehicle.segment_position,
        vehicle.segment_position,
        assigned_speed_mps,
        vehicle,
        [driving_route_part[0]],
        remains_active=True,
    )


def generate_fcds(
    start_time: datetime,
    end_time: datetime,
    start_segment_position: SegmentPosition,
    end_segment_position: SegmentPosition,
    speed: SpeedMps,
    vehicle,
    driving_route_part: List[Segment],
    remains_active: bool,
) -> List[FCDRecord]:
    """
    Generates FCD points on ONE segment for this tick.
    (We clamp movement to segment end in move_on_segment(), so this is consistent.)
    """
    fcds: List[FCDRecord] = []

    seg = driving_route_part[0]
    speed_val = _as_float(speed)
    step_m = speed_val * _as_float(vehicle.fcd_sampling_period / timedelta(seconds=1))

    cur_pos = _as_float(start_segment_position.position)
    cur_time = start_time

    # sample within tick
    while (cur_time + vehicle.fcd_sampling_period < end_time) and (cur_pos + step_m < _as_float(seg.length)):
        cur_pos += step_m
        cur_time += vehicle.fcd_sampling_period
        fcds.append(
            FCDRecord(
                datetime=cur_time,
                vehicle_id=vehicle.id,
                segment=seg,
                offset_from_start=LengthMeters(cur_pos),
                vehicle_speed_mps=SpeedMps(speed_val),
                status=vehicle.status,
                active=True,
                vehicle_type=_norm_vtype(getattr(vehicle, "vehicle_type", "car")),
            )
        )

    # final point at end_time
    fcds.append(
        FCDRecord(
            datetime=end_time,
            vehicle_id=vehicle.id,
            segment=seg,
            offset_from_start=end_segment_position.position,
            vehicle_speed_mps=SpeedMps(speed_val),
            status=vehicle.status,
            active=remains_active,
            vehicle_type=_norm_vtype(getattr(vehicle, "vehicle_type", "car")),
        )
    )
    return fcds


def advance_vehicles_with_queues(
    vehicles_to_be_moved: List,
    departure_time: datetime,
    gv_db: GlobalView,
    routing_map: Map,
    queues_manager: QueuesManager,
    los_vehicles_tolerance,
) -> Tuple[List[FCDRecord], bool]:
    fcds: List[FCDRecord] = []
    vehicles_moved = False

    vehicles_in_queues = {}
    for vehicle in vehicles_to_be_moved:
        queue = queues_manager.queues[(vehicle.current_node, vehicle.next_node)]
        if vehicle.id not in queue:
            prev_pos = vehicle.segment_position
            new_fcds = advance_vehicle(
                vehicle, departure_time, gv_db, routing_map, queues_manager, los_vehicles_tolerance
            )
            fcds.extend(new_fcds)
            vehicles_moved = vehicles_moved or (prev_pos != vehicle.segment_position)
        else:
            vehicles_in_queues[vehicle.id] = vehicle

    for _, queue in queues_manager.queues.copy().items():
        queue_copy = list(queue)
        for vehicle_id in queue_copy:
            if vehicle_id not in vehicles_in_queues:
                break

            vehicle = vehicles_in_queues[vehicle_id]
            del vehicles_in_queues[vehicle_id]

            new_fcds = advance_vehicle(
                vehicle, departure_time, gv_db, routing_map, queues_manager, los_vehicles_tolerance
            )
            fcds.extend(new_fcds)

            was_moved = len(queue) == 0 or (vehicle_id != queue[0])
            vehicles_moved = vehicles_moved or was_moved
            if not was_moved:
                break

    for _, vehicle in vehicles_in_queues.items():
        fcds.extend(advance_waiting_vehicle(vehicle, routing_map, departure_time))

    return fcds, vehicles_moved
