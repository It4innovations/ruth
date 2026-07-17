import hashlib
import logging
import math
from datetime import datetime, timedelta
from typing import List, Tuple

from ..data.map import Map
from ..data.segment import Segment, SegmentPosition, SpeedMps, LengthMeters
from ..globalview_heterogeneous import GlobalView
from .route import MovementInput, MovementModel, MovementResult

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


class HeterogeneousMovementModel(MovementModel):
    def __init__(self, gv_db: GlobalView, routing_map: Map, los_vehicles_tolerance: timedelta):
        self.gv_db = gv_db
        self.routing_map = routing_map
        self.los_vehicles_tolerance = los_vehicles_tolerance

    def compute_batch(self, movement_inputs: List[MovementInput]) -> List[MovementResult]:
        return [
            MovementResult(*move_on_segment(
                movement_input.vehicle,
                movement_input.driving_route_part,
                movement_input.current_time,
                self.gv_db,
                self.routing_map,
                self.los_vehicles_tolerance
            ))
            for movement_input in movement_inputs
        ]


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
