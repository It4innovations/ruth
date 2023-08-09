import logging
import math
from datetime import datetime, timedelta
from typing import List, Tuple

import osmnx as ox

from .queues import QueuesManager
from .segment import Route, Segment, SegmentPosition, SpeedKph
from .simulation import FCDRecord
from ..data.map import Map
from ..losdb import GlobalViewDb
from ..vehicle import Vehicle

logger = logging.getLogger(__name__)


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
    assert segment_position.position <= segment.length

    # if car is stuck in traffic jam, it will not move and its speed will be 0
    if level_of_service == float("inf"):
        # in case the vehicle is stuck in traffic jam just move the time
        return departure_time + vehicle.frequency, segment_position, 0.0

    # Speed in m/s
    speed_mps = (segment.max_allowed_speed_kph * level_of_service) * (1000 / 3600)
    if math.isclose(speed_mps, 0.0):
        return departure_time + vehicle.frequency, segment_position, 0.0

    start_position = segment_position.position
    frequency_s = vehicle.frequency.total_seconds()
    elapsed_m = frequency_s * speed_mps
    end_position = start_position + elapsed_m

    if end_position < segment.length:
        # We stay on the same segment
        return (
            departure_time + timedelta(seconds=frequency_s),
            SegmentPosition(index=segment_position.index, position=end_position),
            speed_mps
        )
    else:
        # The car has finished the segment
        if start_position == segment.length:
            # We have been at the end of the segment the previous round, we will jump to the next segment
            next_segment = segments[segment_position.index + 1]
            if elapsed_m > next_segment.length:
                # if the car makes it to the end of the next segment, it will stay at the end of it
                # travel distance is the length of the next segment
                travel_time = next_segment.length / speed_mps
                return (
                    departure_time + timedelta(seconds=travel_time),
                    SegmentPosition(segment_position.index + 1, next_segment.length),
                    speed_mps
                )
            else:
                return (
                    departure_time + timedelta(seconds=frequency_s),
                    SegmentPosition(segment_position.index + 1, elapsed_m),
                    speed_mps
                )
        else:
            # we just finished the segment, we will stay at the end of it
            travel_distance_m = segment.length - start_position
            travel_time = travel_distance_m / speed_mps
            return (
                departure_time + timedelta(seconds=travel_time),
                SegmentPosition(segment_position.index, segment.length),
                speed_mps
            )


def advance_vehicle(vehicle: Vehicle, departure_time: datetime,
                    gv_db: GlobalViewDb) -> List[FCDRecord]:
    """Advance a vehicle on a route."""

    current_time = departure_time + vehicle.time_offset
    osm_route = vehicle.osm_route

    driving_route = osm_route_to_py_segments(osm_route, vehicle.routing_map)

    fcds = []

    segment = None
    if vehicle.segment_position.index < len(driving_route):
        segment = driving_route[vehicle.segment_position.index]
        offset = vehicle.start_distance_offset
        if vehicle.segment_position.position == segment.length:
            # if the car is at the end of a segment, we want to work with the next segment
            segment = driving_route[vehicle.segment_position.index + 1]
            offset = 0.0
        # los = gv_db.get(current_time, segment)
        los = gv_db.gv.level_of_service_for_car(current_time, segment, vehicle.id, offset)
    else:
        los = 1.0  # the end of the route

    vehicle_end_time, segment_pos, assigned_speed_mps = move_on_segment(
        vehicle, driving_route, current_time, los
    )

    segment_pos_old = vehicle.segment_position
    # NOTE: the segment position index may end out of segments
    if segment_pos.index < len(driving_route):
        fcds = generate_fcds(current_time, vehicle_end_time, segment_pos_old, segment_pos, assigned_speed_mps, vehicle,
                             driving_route)

    # update the vehicle
    vehicle.time_offset += vehicle_end_time - current_time
    vehicle.set_position(segment_pos)

    # step_m = assigned_speed_mps * (vehicle.fcd_sampling_period / timedelta(seconds=1))
    # segment = driving_route[vehicle.segment_position.index]
    # dt = departure_time + vehicle.time_offset
    # logger.info(f"\n{dt} {vehicle.id} ({vehicle.start_distance_offset}) {segment.id} ({segment.length}) step: {step_m}")
    # logger.info(fcds)

    # fill in and out of the queues
    if vehicle.next_node == vehicle.dest_node and vehicle.segment_position.position == segment.length:
        # stop the processing in case the vehicle reached the end
        vehicle.active = False

    elif vehicle.segment_position.position == segment.length:
        QueuesManager.add_to_queue(vehicle)

    segment_old = driving_route[segment_pos_old.index]
    if segment_pos_old.position == segment_old.length and segment_pos_old.index != vehicle.segment_position.index:
        node_from, node_to = vehicle.osm_route[segment_pos_old.index], vehicle.osm_route[segment_pos_old.index + 1]
        QueuesManager.remove_vehicle(vehicle, node_from, node_to)

    return fcds


def advance_waiting_vehicle(vehicle: Vehicle, departure_time: datetime) -> List[FCDRecord]:
    current_time = departure_time + vehicle.time_offset
    osm_route = vehicle.osm_route

    driving_route = osm_route_to_py_segments(osm_route, vehicle.routing_map)
    segment_pos_old = vehicle.segment_position
    vehicle_end_time, segment_pos, assigned_speed_mps = move_on_segment(
        vehicle, driving_route, current_time, float("inf")
    )

    # update the vehicle
    vehicle.time_offset += vehicle_end_time - current_time

    fcds = generate_fcds(current_time, vehicle_end_time, segment_pos_old, segment_pos, assigned_speed_mps, vehicle,
                         driving_route)
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


def generate_fcds(start_time: datetime, end_time: datetime, start_segment_position: SegmentPosition,
                  end_segment_position: SegmentPosition, speed: SpeedKph, vehicle: Vehicle,
                  driving_route: list[Segment]) -> List[FCDRecord]:
    fcds = []

    step_m = speed * (vehicle.fcd_sampling_period / timedelta(seconds=1))

    # when both start and end positions are on the same segment
    current_position = start_segment_position.position
    current_time = start_time
    current_segment = driving_route[start_segment_position.index]

    if current_position == current_segment.length:
        # when the vehicle finished the segment in the previous round, we will jump to the next segment
        current_segment = driving_route[end_segment_position.index]
        current_position = 0

    while current_time + vehicle.fcd_sampling_period < end_time and \
            current_position + step_m < current_segment.length:
        current_position += step_m
        current_time += vehicle.fcd_sampling_period
        fcds.append(FCDRecord(
            datetime=current_time,
            vehicle_id=vehicle.id,
            segment_id=current_segment.id,
            segment_length=current_segment.length,
            start_offset=current_position,
            speed=speed,
            status=vehicle.status
        ))

    # store end of the movement
    fcds.append(FCDRecord(
        datetime=end_time,
        vehicle_id=vehicle.id,
        segment_id=current_segment.id,
        segment_length=current_segment.length,
        start_offset=end_segment_position.position,
        speed=speed,
        status=vehicle.status
    ))
    return fcds
