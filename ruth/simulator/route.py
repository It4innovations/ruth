import logging
import math
from datetime import datetime, timedelta
from typing import List, Tuple

from .queues import QueuesManager
from ..data.map import Map
from ..data.segment import Segment, SegmentPosition, SpeedMps, LengthMeters, speed_kph_to_mps
from .simulation import FCDRecord
from ..losdb import GlobalViewDb
from ..vehicle import Vehicle

logger = logging.getLogger(__name__)


def move_on_segment(
        vehicle: Vehicle,
        driving_route: List[Segment],
        current_time: datetime,
        gv_db: GlobalViewDb,
        routing_map: Map,
        los_vehicles_tolerance: timedelta = timedelta(seconds=0)
) -> Tuple[datetime, SegmentPosition, SpeedMps]:
    """
    Moves the car on its current segment.
    Returns (time, position, speed) at the end of the movement.
    """
    segment_position = vehicle.segment_position
    start_position = segment_position.position
    current_segment = driving_route[segment_position.index]
    assert segment_position.position <= current_segment.length

    if vehicle.segment_position.index < len(driving_route) and start_position == current_segment.length:
        # if the vehicle is at the end of a segment and there are more segments in the route
        if vehicle.has_next_segment_closed(routing_map):
            return current_time + vehicle.frequency, vehicle.segment_position, SpeedMps(0.0)
        # if the vehicle can move to the next segment, work with the next segment
        start_position = LengthMeters(0.0)
        segment_position = SegmentPosition(segment_position.index + 1, start_position)
        current_segment = driving_route[segment_position.index]

    if segment_position.index == len(driving_route) and start_position == current_segment.length:
        # the end of the driving route
        level_of_service = 1.0
    else:
        level_of_service = gv_db.gv.level_of_service_in_front_of_vehicle(current_time, current_segment, vehicle.id,
                                                                         start_position, los_vehicles_tolerance)

    # if car is stuck in traffic jam, it will not move and its speed will be 0
    if level_of_service == float("inf"):
        # in case the vehicle is stuck in traffic jam just move the time
        return current_time + vehicle.frequency, vehicle.segment_position, SpeedMps(0.0)

    # Speed in m/s
    speed_mps = speed_kph_to_mps(current_segment.max_allowed_speed_kph * level_of_service)
    if math.isclose(speed_mps, 0.0):
        # in case the vehicle is not moving, move the time and keep the previous position
        return current_time + vehicle.frequency, vehicle.segment_position, SpeedMps(0.0)

    frequency_s = vehicle.frequency.total_seconds()
    elapsed_m = frequency_s * speed_mps
    end_position = LengthMeters(start_position + elapsed_m)

    if end_position < current_segment.length:
        return (
            current_time + timedelta(seconds=frequency_s),
            SegmentPosition(index=segment_position.index, position=end_position),
            speed_mps
        )
    else:
        # The car has finished the segment, it stays at the end of it
        travel_distance_m = current_segment.length - start_position
        travel_time = travel_distance_m / speed_mps
        return (
            current_time + timedelta(seconds=travel_time),
            SegmentPosition(segment_position.index, current_segment.length),
            speed_mps
        )


def advance_vehicle(vehicle: Vehicle, departure_time: datetime,
                    gv_db: GlobalViewDb, routing_map: Map, queues_manager: QueuesManager,
                    los_vehicles_tolerance: timedelta = timedelta(seconds=0)) -> List[FCDRecord]:
    """Advance a vehicle on a route."""

    current_time = departure_time + vehicle.time_offset
    fcds = []

    osm_route = vehicle.osm_route
    driving_route = routing_map.osm_route_to_py_segments(osm_route)

    vehicle_end_time, segment_pos, assigned_speed_mps = move_on_segment(
        vehicle, driving_route, current_time, gv_db, routing_map, los_vehicles_tolerance
    )

    segment_pos_old = vehicle.segment_position

    # update the vehicle
    vehicle.time_offset += vehicle_end_time - current_time
    vehicle.set_position(segment_pos)

    # step_m = assigned_speed_mps * (vehicle.fcd_sampling_period / timedelta(seconds=1))
    # segment = driving_route[vehicle.segment_position.index]
    # dt = departure_time + vehicle.time_offset
    # logger.info(f"\n{dt} {vehicle.id} ({vehicle.start_distance_offset}) {segment.id} ({segment.length}) step: {step_m}")
    # logger.info(fcds)

    segment = driving_route[segment_pos.index]
    segment_old = driving_route[segment_pos_old.index]

    # fill in and out of the queues
    if (segment_pos_old.position == segment_old.length
            and segment_pos_old.index != vehicle.segment_position.index):
        # remove vehicle from outdated queue if it changed segment
        node_from, node_to = vehicle.osm_route[segment_pos_old.index], vehicle.osm_route[segment_pos_old.index + 1]
        queues_manager.remove_vehicle(vehicle, node_from, node_to)

    if vehicle.start_distance_offset == segment.length:
        if vehicle.next_node == vehicle.dest_node:
            # stop the processing in case the vehicle reached the end
            vehicle.active = False
            queues_manager.remove_inactive_vehicle(vehicle)

        elif segment_pos_old != vehicle.segment_position:
            # if the vehicle is at the end of the segment and was not there before, add it to the queue
            queues_manager.add_to_queue(vehicle)

    # NOTE: the segment position index may end out of segments
    if segment_pos.index < len(driving_route):
        fcds = generate_fcds(current_time, vehicle_end_time, segment_pos_old, segment_pos, assigned_speed_mps, vehicle,
                             driving_route, remains_active=vehicle.active)

    return fcds


def advance_waiting_vehicle(vehicle: Vehicle, routing_map: Map, departure_time: datetime) -> List[FCDRecord]:
    current_time = departure_time + vehicle.time_offset
    osm_route = vehicle.osm_route

    driving_route = routing_map.osm_route_to_py_segments(osm_route)
    segment_pos_old = vehicle.segment_position

    # in case the vehicle is not moving, move the time and keep the previous position
    vehicle_end_time = current_time + vehicle.frequency
    segment_pos = vehicle.segment_position
    assigned_speed_mps = SpeedMps(0.0)

    # update the vehicle
    vehicle.time_offset += vehicle_end_time - current_time

    fcds = generate_fcds(current_time, vehicle_end_time, segment_pos_old, segment_pos, assigned_speed_mps, vehicle,
                         driving_route, remains_active=True)
    return fcds


def generate_fcds(start_time: datetime, end_time: datetime, start_segment_position: SegmentPosition,
                  end_segment_position: SegmentPosition, speed: SpeedMps, vehicle: Vehicle,
                  driving_route: List[Segment], remains_active: bool) -> List[FCDRecord]:
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
            segment=current_segment,
            start_offset=current_position,
            speed=speed,
            status=vehicle.status,
            active=True
        ))

    # store end of the movement
    fcds.append(FCDRecord(
        datetime=end_time,
        vehicle_id=vehicle.id,
        segment=current_segment,
        start_offset=end_segment_position.position,
        speed=speed,
        status=vehicle.status,
        active=remains_active
    ))
    return fcds


def advance_vehicles_with_queues(vehicles_to_be_moved: List[Vehicle], departure_time: datetime,
                                   gv_db: GlobalViewDb, routing_map: Map, queues_manager: QueuesManager,
                                   los_vehicles_tolerance) -> Tuple[List[FCDRecord], bool]:
    fcds = []

    vehicles_moved = False
    vehicles_in_queues = []
    for vehicle in vehicles_to_be_moved:
        queue = queues_manager.queues[(vehicle.current_node, vehicle.next_node)]
        if vehicle not in queue:
            new_fcds = advance_vehicle(vehicle, departure_time, gv_db, routing_map, queues_manager,
                                       los_vehicles_tolerance)
            fcds.extend(new_fcds)
            vehicles_moved = True
        else:
            vehicles_in_queues.append(vehicle)

    for key, queue in queues_manager.queues.copy().items():
        for vehicle in queue.copy():
            try:
                vehicles_in_queues.remove(vehicle)
            except ValueError:
                break

            new_fcds = advance_vehicle(vehicle, departure_time, gv_db, routing_map, queues_manager,
                                       los_vehicles_tolerance)
            fcds.extend(new_fcds)
            was_moved = len(queue) == 0 or (vehicle != queue[0])
            vehicles_moved = vehicles_moved or was_moved
            if not was_moved:
                break

    for vehicle in vehicles_in_queues:
        new_fcds = advance_waiting_vehicle(vehicle, routing_map, departure_time)
        fcds.extend(new_fcds)

    assert len(fcds) == len(vehicles_to_be_moved)
    return fcds, vehicles_moved
