import math
from datetime import datetime, timedelta
from typing import List, Tuple

import osmnx as ox

from .segment import Route, Segment, SegmentPosition, SpeedKph
from ..data.map import Map
from ..losdb import GlobalViewDb
from ..vehicle import Vehicle


def move_on_segment(
        vehicle: Vehicle,
        segments: List[Segment],
        departure_time: datetime,
        level_of_service: float
) -> Tuple[datetime, SegmentPosition, SpeedKph]:
    """
    Moves the car on its current segment.
    Returns (time, position, speed) at the end of the movement.
    """
    segment_position = vehicle.segment_position
    segment = segments[segment_position.index]
    assert segment_position.position < segment.length

    # Speed in m/s
    speed_mps = (segment.max_allowed_speed_kph * level_of_service) * (1000 / 3600)
    if math.isclose(speed_mps, 0.0):
        return (departure_time, segment_position, 0.0)

    start = segment_position.position
    frequency_s = vehicle.frequency.total_seconds()
    elapsed_m = start + frequency_s * speed_mps

    if elapsed_m < segment.length:
        # We stay on the same segment
        return (
            departure_time + timedelta(seconds=frequency_s),
            SegmentPosition(index=segment_position.index, position=elapsed_m),
            speed_mps
        )
    else:
        # We have moved to the end of the current segment, the car will jump to the
        # beginning of the next one.
        travel_distance_m = segment.length - start
        travel_time = travel_distance_m / speed_mps
        return (
            departure_time + timedelta(seconds=travel_time),
            SegmentPosition(segment_position.index + 1, 0.0),
            speed_mps
        )


def advance_vehicle(vehicle: Vehicle, departure_time: datetime,
                    gv_db: GlobalViewDb):
    """Advance a vehicle on a route."""

    dt = departure_time + vehicle.time_offset
    osm_route = vehicle.osm_route

    driving_route = osm_route_to_py_segments(osm_route, vehicle.routing_map)

    if vehicle.segment_position.index < len(driving_route):
        segment = driving_route[vehicle.segment_position.index]
        los = gv_db.get(dt, segment)
    else:
        los = 1.0  # the end of the route

    if los == float("inf"):
        # in case the vehicle is stuck in traffic jam just move the time
        vehicle.time_offset += vehicle.frequency
    else:
        time, segment_pos, assigned_speed_mps = move_on_segment(
            vehicle, driving_route, dt, los
        )
        d = time - dt

        # NOTE: _assumption_: the car stays on a single segment within one call of the `advance`
        #       method on the driving route

        # NOTE: the segment position index may end out of segments
        if segment_pos.index < len(driving_route):
            vehicle.store_fcd(dt, d, driving_route[segment_pos.index], segment_pos.position,
                              assigned_speed_mps)

        # update the vehicle
        vehicle.time_offset += d
        vehicle.set_position(segment_pos)

        if vehicle.current_node == vehicle.dest_node:
            # stop the processing in case the vehicle reached the end
            vehicle.active = False

    return vehicle


def osm_route_to_py_segments(osm_route: Route, routing_map: Map) -> List[Segment]:
    """Prepare list of segments based on route."""
    edge_data = ox.utils_graph.get_route_edge_attributes(routing_map.network,
                                                         osm_route)
    edges = zip(osm_route, osm_route[1:])
    return [
        Segment(
            id=f"OSM{from_}T{to_}",
            length=data["length"],
            max_allowed_speed_kph=data["speed_kph"],
        )
        # NOTE: the zip is correct as the starting node_id is of the interest
        for i, ((from_, to_), data) in enumerate(zip(edges, edge_data))
    ]
