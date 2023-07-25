import math
from datetime import datetime, timedelta
from typing import List, Tuple

import osmnx as ox

from .segment import Route, Segment, SegmentPosition, SpeedKph
from .simulation import FCDRecord
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
                    gv_db: GlobalViewDb) -> List[FCDRecord]:
    """Advance a vehicle on a route."""

    dt = departure_time + vehicle.time_offset
    osm_route = vehicle.osm_route

    driving_route = osm_route_to_py_segments(osm_route, vehicle.routing_map)

    fcds = []

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
            fcds = generate_fcds(vehicle, dt, d, driving_route[segment_pos.index], segment_pos.position,
                              assigned_speed_mps)

        # update the vehicle
        vehicle.time_offset += d
        vehicle.set_position(segment_pos)

        if vehicle.current_node == vehicle.dest_node:
            # stop the processing in case the vehicle reached the end
            vehicle.active = False

    return fcds


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


def generate_fcds(vehicle: Vehicle, start_offset: datetime, duration: timedelta, segment: Segment,
                  start_position: float, speed: float) -> List[FCDRecord]:
    fcds = []

    step_m = speed * (vehicle.fcd_sampling_period / timedelta(seconds=1))

    start = start_position
    current_offset = start_offset
    end_offset = start_offset + duration
    while current_offset + vehicle.fcd_sampling_period < end_offset and \
            start + step_m < segment.length:
        start += step_m
        current_offset += vehicle.fcd_sampling_period
        fcds.append(FCDRecord(
            datetime=current_offset,
            vehicle_id=vehicle.id,
            segment_id=segment.id,
            segment_length=segment.length,
            start_offset=start,
            speed=speed,
            status=vehicle.status
        ))

    step_m = speed * ((end_offset - current_offset) / timedelta(seconds=1))
    if start + step_m < segment.length:
        # TODO: the question is wheather to store all the cars at the end of period or
        # rather return the difference (end_offset - _last_ current_offset) and take it as
        # a parameter for the next round of storing. In this way all the cars would be sampled
        # with an exact step (car dependent as each car can have its own sampling period)
        fcds.append(FCDRecord(
            datetime=end_offset,
            vehicle_id=vehicle.id,
            segment_id=segment.id,
            segment_length=segment.length,
            start_offset=start + step_m,
            speed=speed,
            status=vehicle.status
        ))
    return fcds
