import logging
from datetime import datetime, timedelta
from typing import Generator, List, NamedTuple, Tuple

from .queues import QueuesManager
from ..data.map import Map
from ..data.segment import Segment, SegmentPosition, SpeedMps, LengthMeters, speed_kph_to_mps
from .simulation import FCDRecord
from ..globalview import GlobalView
from ..vehicle import Vehicle

logger = logging.getLogger(__name__)


class MovementInput(NamedTuple):
    vehicle: Vehicle
    current_time: datetime
    driving_route_part: List[Segment]


class MovementResult(NamedTuple):
    vehicle_end_time: datetime
    segment_pos: SegmentPosition
    assigned_speed_mps: SpeedMps


class MovementModel:
    def compute_batch(self, movement_inputs: List[MovementInput]) -> List[MovementResult]:
        raise NotImplementedError


class DefaultMovementModel(MovementModel):
    def __init__(self, gv_db: GlobalView, routing_map: Map, los_vehicles_tolerance: timedelta):
        self.gv_db = gv_db
        self.routing_map = routing_map
        self.los_vehicles_tolerance = los_vehicles_tolerance

    def compute_batch(self, movement_inputs: List[MovementInput]) -> List[MovementResult]:
        results = []
        for movement_input in movement_inputs:
            speed_mps, changed_segment = get_vehicle_speed(
                movement_input.vehicle,
                movement_input.driving_route_part,
                movement_input.current_time,
                self.gv_db,
                self.routing_map,
                self.los_vehicles_tolerance
            )
            results.append(MovementResult(*move_on_segment(
                movement_input.vehicle,
                movement_input.driving_route_part,
                movement_input.current_time,
                speed_mps,
                changed_segment
            )))
        return results


def get_vehicle_inputs(vehicles: List[Vehicle], departure_time: datetime,
                       routing_map: Map) -> Generator[MovementInput, None, None]:
    """
    Generator that yields vehicles with their current time and route information.
    Performance optimized: processes vehicles efficiently without storing intermediate lists.

    Args:
        vehicles: List of vehicles to process
        departure_time: Simulation departure time
        routing_map: Map with routing info

    Yields:
        MovementInput(vehicle, current_time, driving_route_part)
    """
    for vehicle in vehicles:
        current_vehicle_index = vehicle.start_index
        osm_route_part = vehicle.osm_route[current_vehicle_index:current_vehicle_index + 3]
        driving_route_part = routing_map.osm_route_to_py_segments(osm_route_part)

        yield MovementInput(vehicle, departure_time + vehicle.time_offset, driving_route_part)


def get_input(departure_time: datetime, vehicle: Vehicle, routing_map: Map) -> MovementInput:
    """
    Get movement input for a vehicle.

    Returns:
        MovementInput(vehicle, current_time, driving_route_part)
    """
    current_vehicle_index = vehicle.start_index
    osm_route_part = vehicle.osm_route[current_vehicle_index:current_vehicle_index + 3]
    driving_route_part = routing_map.osm_route_to_py_segments(osm_route_part)

    return MovementInput(vehicle, departure_time + vehicle.time_offset, driving_route_part)


def get_vehicle_speed(
        vehicle: Vehicle,
        driving_route_part: List[Segment],
        current_time: datetime,
        gv_db: GlobalView,
        routing_map: Map,
        los_vehicles_tolerance: timedelta = timedelta(seconds=0)) -> Tuple[SpeedMps, bool]:
    """
    Calculate vehicle speed based on level of service.

    Returns:
        (speed_mps, changed_segment)
    """
    segment_position = vehicle.segment_position
    start_position = segment_position.position
    current_segment = driving_route_part[0]
    assert segment_position.position <= current_segment.length
    changed_segment = False

    if start_position == current_segment.length:
        # if the vehicle is at the end of a segment and there are more segments in the route
        if vehicle.has_next_segment_closed(routing_map):
            return SpeedMps(0.0), False
        # if the vehicle can move to the next segment, work with the next segment
        start_position = LengthMeters(0.0)
        changed_segment = True
        current_segment = driving_route_part[1]

    level_of_service = gv_db.level_of_service_in_front_of_vehicle(current_time, current_segment, vehicle.id,
                                                                  start_position, los_vehicles_tolerance)

    # if car is stuck in traffic jam, it will not move and its speed will be 0
    if level_of_service == float("inf"):
        # in case the vehicle is stuck in traffic jam just move the time
        return SpeedMps(0.0), False

    # Speed in m/s
    speed_mps = speed_kph_to_mps(current_segment.max_allowed_speed_kph * level_of_service)
    return speed_mps, changed_segment

def move_on_segment(
        vehicle: Vehicle,
        driving_route_part: List[Segment],
        current_time: datetime,
        speed_mps: SpeedMps,
        changed_segment: bool,
        segment_end_delta=2
) -> Tuple[datetime, SegmentPosition, SpeedMps]:
    """
    Moves the car on its current segment.
    Returns (time, position, speed) at the end of the movement.
    """
    if not changed_segment:
        current_segment = driving_route_part[0]
        segment_position = vehicle.segment_position
    else:
        current_segment = driving_route_part[1]
        segment_position = SegmentPosition(
            index=vehicle.segment_position.index + 1,
            position=LengthMeters(0.0)
        )
    start_position = segment_position.position

    if speed_mps == 0.0:
        return current_time + vehicle.frequency, vehicle.segment_position, SpeedMps(0.0)

    frequency_s = vehicle._frequency_seconds # type: ignore
    elapsed_m = frequency_s * speed_mps
    end_position = LengthMeters(start_position + elapsed_m)

    segment_length = current_segment.length

    # Check if vehicle reaches or passes segment end
    if end_position >= (segment_length - segment_end_delta):
        # Vehicle finishes segment
        # (Segment length exactly, more or near the end)
        travel_distance_m = segment_length - start_position
        travel_time = travel_distance_m / speed_mps
        return (
            current_time + timedelta(seconds=travel_time),
            SegmentPosition(index=segment_position.index, position=segment_length),
            speed_mps
        )
    else:
        # Vehicle continues within segment
        return (
            current_time + timedelta(seconds=frequency_s),
            SegmentPosition(index=segment_position.index, position=end_position),
            speed_mps
        )


def advance_vehicle(movement_input: MovementInput, movement_result: MovementResult,
                    queues_manager: QueuesManager) -> List[FCDRecord]:
    """Advance a vehicle on a route."""

    vehicle = movement_input.vehicle
    current_time = movement_input.current_time
    driving_route_part = movement_input.driving_route_part
    fcds = []

    vehicle_end_time = movement_result.vehicle_end_time
    segment_pos = movement_result.segment_pos
    assigned_speed_mps = movement_result.assigned_speed_mps

    segment_pos_old = vehicle.segment_position

    # check vehicle's first move
    if segment_pos.index == 0 and segment_pos.position == 0.0:
        # vehicle could not move on segment, skip this round and just advance the time
        vehicle.time_offset += vehicle.frequency
        return []

    # update the vehicle
    vehicle.time_offset += vehicle_end_time - current_time
    vehicle.set_position(segment_pos)

    segment = driving_route_part[segment_pos.index - segment_pos_old.index]
    segment_old = driving_route_part[0]

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

        elif segment_pos_old != vehicle.segment_position:
            # if the vehicle is at the end of the segment and was not there before, add it to the queue
            queues_manager.add_to_queue(vehicle)

    # NOTE: the segment position index may end out of segments
    if segment_pos.index >= (len(vehicle.osm_route)):
       print(f"Vehicle {vehicle.id} segment index {segment_pos.index} out of route bounds {len(vehicle.osm_route)}")
       print(vehicle)

    fcds = generate_fcds(current_time, vehicle_end_time, segment_pos_old, segment_pos, assigned_speed_mps, vehicle,
                             driving_route_part, remains_active=vehicle.active)

    return fcds


def advance_waiting_vehicle(vehicle: Vehicle, routing_map: Map, departure_time: datetime) -> List[FCDRecord]:
    current_time = departure_time + vehicle.time_offset

    current_vehicle_index = vehicle.start_index
    osm_route_part = vehicle.osm_route[current_vehicle_index:current_vehicle_index + 3]
    driving_route_part = routing_map.osm_route_to_py_segments(osm_route_part)

    # in case the vehicle is not moving, move the time and keep the previous position
    vehicle_end_time = current_time + vehicle.frequency
    segment_pos = vehicle.segment_position
    assigned_speed_mps = SpeedMps(0.0)

    # update the vehicle
    vehicle.time_offset += vehicle_end_time - current_time

    fcds = generate_fcds(current_time, vehicle_end_time, segment_pos, segment_pos, assigned_speed_mps, vehicle,
                         driving_route_part, remains_active=True)
    return fcds


def generate_fcds(start_time: datetime, end_time: datetime, start_segment_position: SegmentPosition,
                  end_segment_position: SegmentPosition, speed: SpeedMps, vehicle: Vehicle,
                  driving_route_part: List[Segment], remains_active: bool) -> List[FCDRecord]:
    fcds = []

    step_m = speed * (vehicle.fcd_sampling_period / timedelta(seconds=1))

    # when both start and end positions are on the same segment
    current_position = start_segment_position.position
    current_time = start_time
    current_segment = driving_route_part[0]

    if current_position == current_segment.length and start_segment_position.index != end_segment_position.index:
        # when the vehicle finished the segment in the previous round, we will jump to the next segment
        current_segment = driving_route_part[1]
        current_position = 0

    while current_time + vehicle.fcd_sampling_period < end_time and \
            current_position + step_m < current_segment.length:
        current_position += step_m
        current_time += vehicle.fcd_sampling_period
        fcds.append(FCDRecord(
            datetime=current_time,
            vehicle_id=vehicle.id,
            segment=current_segment,
            offset_from_start=current_position,
            vehicle_speed_mps=speed,
            status=vehicle.status,
            active=True
        ))

    # store end of the movement
    fcds.append(FCDRecord(
        datetime=end_time,
        vehicle_id=vehicle.id,
        segment=current_segment,
        offset_from_start=end_segment_position.position,
        vehicle_speed_mps=speed,
        status=vehicle.status,
        active=remains_active
    ))
    return fcds


def advance_vehicles_with_queues(vehicles_to_be_moved: List[Vehicle], departure_time: datetime,
                                 gv_db: GlobalView, routing_map: Map, queues_manager: QueuesManager,
                                 los_vehicles_tolerance,
                                 movement_model=None) -> Tuple[List[FCDRecord], bool]:
    fcds = []

    if movement_model is None:
        movement_model = DefaultMovementModel(gv_db, routing_map, los_vehicles_tolerance)

    vehicles_moved = False
    vehicles_in_queues = dict()
    vehicles_not_in_queues = []

    # SPLIT VEHICLES IN QUEUES AND NOT IN QUEUES
    for vehicle in vehicles_to_be_moved:
        queue = queues_manager.queues[(vehicle.current_node, vehicle.next_node)]
        if vehicle.id not in queue:
            vehicles_not_in_queues.append(vehicle)
        else:
            vehicles_in_queues[vehicle.id] = vehicle

    # Process vehicles not in queues
    movement_inputs = list(get_vehicle_inputs(vehicles_not_in_queues, departure_time, routing_map))
    movement_results = movement_model.compute_batch(movement_inputs)
    for movement_input, movement_result in zip(movement_inputs, movement_results):
        prev_pos = movement_input.vehicle.segment_position
        new_fcds = advance_vehicle(movement_input, movement_result, queues_manager)
        if new_fcds:
            fcds.extend(new_fcds)
        vehicles_moved = vehicles_moved or prev_pos != movement_input.vehicle.segment_position

    # --------------------------------------------------------------------------------------------------------------
    # MOVE VEHICLES IN QUEUES
    processed_ids = set()
    for _, queue in queues_manager.queues.items():
        for vehicle_id in queue:
            if vehicle_id not in vehicles_in_queues:
                break

            vehicle = vehicles_in_queues[vehicle_id]
            processed_ids.add(vehicle_id)

            prev_pos = vehicle.segment_position
            movement_input = get_input(departure_time, vehicle, routing_map)
            movement_result = movement_model.compute_batch([movement_input])[0]
            new_fcds = advance_vehicle(movement_input, movement_result, queues_manager)
            if new_fcds:
                fcds.extend(new_fcds)

            was_moved = prev_pos != vehicle.segment_position
            vehicles_moved = vehicles_moved or was_moved
            if not was_moved:
                break

    # Process unprocessed vehicles in queues (waiting vehicles)
    for vehicle_id, vehicle in vehicles_in_queues.items():
        if vehicle_id in processed_ids:
            continue
        new_fcds = advance_waiting_vehicle(vehicle, routing_map, departure_time)
        if new_fcds:
            fcds.extend(new_fcds)

    queues_manager.batch_update()
    return fcds, vehicles_moved
