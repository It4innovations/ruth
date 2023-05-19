
import logging
import pandas as pd
from dataclasses import asdict
from datetime import timedelta
from itertools import product

from probduration import HistoryHandler, Route, SegmentPosition, probable_delay

from ..utils import osm_route_to_segments
from ..vehicle import Vehicle


logger = logging.getLogger(__name__)


class ParamsGenerator:
    def __init__(self):
        self.params = dict()

    def register(self, name, iterable):
        assert name not in self.params, f"Parameter '{name}' is already registered."
        self.params[name] = iterable

    def drop(self, name):
        del self.params[name]

    def __iter__(self):
        prod = product(*self.params.values())
        keys = self.params.keys()

        for comb in prod:
            yield dict(zip(keys, comb))

    def __len__(self):
        return len(list(product(*self.params.values())))


def load_vehicles(input_path: str):
    logger.info("Loading data ... %s", input_path)
    df = pd.read_parquet(input_path, engine="fastparquet")
    return [Vehicle(**row.to_dict()) for (_, row) in df.iterrows()]


def save_vehicles(vehicles, output_path: str):
    logger.info("Saving vehicles ... %s", output_path)

    df = pd.DataFrame([asdict(v) for v in vehicles])
    df.to_pickle(output_path)


def advance_vehicle(vehicle, osm_route, departure_time, gv_db):
    """Advance a vehicle on a route."""

    dt = departure_time + vehicle.time_offset

    driving_route = Route(osm_route_to_segments(osm_route, vehicle.routing_map),
                          vehicle.frequency)
    # update the current route
    vehicle.set_current_route(osm_route)

    if vehicle.segment_position.index < len(driving_route):
        segment = driving_route[vehicle.segment_position.index]
        los = gv_db.get(dt, segment)
    else:
        los = 1.0  # the end of the route

    if los == float("inf"):
        # in case the vehicle is stuck in traffic jam just move the time
        vehicle.time_offset += vehicle.frequency
    else:
        time, segment_pos, assigned_speed_mps = driving_route.advance_with_speed(
            vehicle.segment_position, dt, los)
        d = time - dt

        # NOTE: _assumption_: the car stays on a single segment within one call of the `advance`
        #       method on the driving route

        if segment_pos.index < len(driving_route):  # NOTE: the segment position index may end out of segments
            vehicle.store_fcd(dt, d, driving_route[segment_pos.index], segment_pos.start, assigned_speed_mps)

        # update the vehicle
        vehicle.time_offset += d
        vehicle.set_position(segment_pos)

        if vehicle.current_node == vehicle.dest_node:
            # stop the processing in case the vehicle reached the end
            vehicle.active = False

    return vehicle


def distance_duration(driving_route, departure_time, los_db, rnd_gen, stop_distance=None):

    distance = 0.0
    p = SegmentPosition(0, 0.0)
    dt = departure_time
    level_of_services = []

    stop_distance_ = stop_distance if stop_distance is not None else driving_route.distance_in_meters()

    while distance < stop_distance_:
        if p.index >= len(driving_route):
            break

        seg = driving_route[p.index]
        los = los_db.get(dt, seg, rnd_gen())

        if los == float('inf'):  # stuck in traffic jam; not moving
            return float('inf'), None, None

        elapsed, next_segment_pos, assigned_speed_mps = driving_route.advance(p, dt, los)
        d = elapsed - dt

        if p.index == next_segment_pos.index:
            # movement on the same segment
            distance += next_segment_pos.start - p.start
        else:
            # if the next segment is different, i.e. its index is higher than
            # the rest distance of the previous segment is added.
            distance += seg.length - p.start

        if distance > stop_distance_:
            # round down to the stop distance

            # decrease the distance
            dd = distance - stop_distance_
            if next_segment_pos.start - dd < 0:
                next_segment_pos = SegmentPosition(p.index, seg.length + (next_segment_pos.start - dd))
            else:
                next_segment_pos.start -= dd

            # decrease the duration by overpass time
            over_duration = dd / assigned_speed_mps
            d -= timedelta(seconds=over_duration)

        level_of_services.append(los)
        dt += d
        p = next_segment_pos

    # returns:
    # duration, next segment position, average level of service
    if len(level_of_services) == 0:
        return float("inf"), None, None

    avg_los = sum(level_of_services) / len(level_of_services)

    return dt - departure_time, p, avg_los


def alternatives(vehicle, k):
    return vehicle.k_shortest_paths(k)
