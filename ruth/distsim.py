"""A distributed traffic simulator."""

import logging
import os
import random
import pandas as pd
import dask.bag as db
from copy import copy
from dask.distributed import Client
from dataclasses import asdict
from datetime import timedelta

from probduration import HistoryHandler, Route, SegmentPosition, probable_duration

from ruth.utils import osm_route_to_segments
from ruth.vehicle import Vehicle
from ruth.globalview import GlobalView


logger = logging.getLogger(__name__)


def load_vehicles(input_path: str):
    logger.info("Loading data ... %s", input_path)
    df = pd.read_pickle(input_path)
    return [Vehicle(**row.to_dict()) for (_, row) in df.iterrows()]


def simulate(input_path: str,
             scheduler: str,
             scheduler_port: int,
             departure_time,
             k_routes,
             n_samples,
             seed,
             gv_update_period,
             intermediate_results,
             checkpoint_period):
    """Distributed traffic simulator."""

    if seed is not None:
        random.seed(seed)  # used for tests: 660277

    if intermediate_results is not None:
        intermediate_results = os.path.abspath(sintermediate_results)

        if not os.path.exists(intermediate_results):
            os.mkdir(intermediate_results)

    gv = GlobalView()
    c = Client(f"{scheduler}:{scheduler_port}")
    dist_gv = c.scatter(gv, broadcast=True)

    vehicles = db.from_sequence(load_vehicles(input_path))

    step = 0
    while True:
        logger.info("Starting step %s", step)

        def compute_min(partitions): # TODO: why the method is created in each cycle?
            return min((v.time_offset for v in partitions), default=float("inf"))

        min_offset = vehicles.reduction(compute_min, min).compute()

        def gather_active(partitions):
            return any(v.active for v in partitions)

        active = vehicles.reduction(gather_active, any).compute()

        if not active:
            # No active cars
            break

        def advance(vehicle, gv):
            if vehicle.time_offset == min_offset:
                gv_near_distance = 200 # TODO: make it as parameter
                return advance_vehicle(vehicle, departure_time, k_routes, gv, gv_near_distance, n_samples)
            else:
                return vehicle

        vehicles = vehicles.map(advance, dist_gv).persist()

        if intermediate_results is not None and step % (checkpoint_period * gv_update_periodl) == 0:
            # TODO: Save vehicles
            pass

        def process_leap_history(partitions):
            leap_history_update = []
            for vehicle in partitions:
                if vehicle.leap_history:
                    # process only non-empty history
                    leap_history_update.append((vehicle.id, vehicle.leap_history[:]))
            return leap_history_update

        def join_leap_histories(lh_updates):
            leap_history = []
            for lh_update in lh_updates:
                leap_history += lh_update
            return leap_history

        # leap_history_update = vehicles.reduction(
        #     process_leap_history,
        #     join_leap_histories
        # ).compute()

        # if leap_history_update: # update only if there is data
        #     for vehicle_id, lhu in leap_history_update:
        #         gv.add(vehicle_id, lhu)

        #     dist_gv = c.scatter(gv, broadcast=True)

        def clear_leap_history(vehicle):
            data = asdict(vehicle)
            data["leap_history"] = []
            return Vehicle(**data)

        vehicles = vehicles.map(clear_leap_history).persist()

        step += 1

    return (gv, vehicles)


def advance_vehicle(vehicle, departure_time, k_routes, gv, gv_distance, nsamples=1):
    """Advance a vehicle on a route."""

    dt = departure_time + vehicle.time_offset
    prob_profiles = HistoryHandler.no_limit()  # TODO: history get here or take as an argument?

    # advance the vehicle on the driving route
    osm_route, driving_route = ptdr(vehicle, dt, k_routes, gv, gv_distance, prob_profiles, nsamples)

    # update the current route
    vehicle.set_current_route(osm_route)


    if vehicle.segment_position.index < len(driving_route):
        segment = driving_route[vehicle.segment_position.index]
        los = gv.level_of_service_in_time_at_segment(dt, segment, tolerance=timedelta(seconds=15))
    else:
        los = 1.0  # the end of the route

    if los == float("inf"):
        # in case the vehicle is stuck in traffic jam just move the time
        time = dt + vehicle.frequency
        vehicle.time_offset += vehicle.frequency
    else:
        time, segment_pos = driving_route.advance(
            vehicle.segment_position, dt, los)
        d = time - dt

        # NOTE: _assumtion_: the car stays on a single segment within one call of the `advance`
        #       method on the driving route

        if segment_pos.index < len(driving_route):  # NOTE: the segment position index may end out of segments
            vehicle.store_fcd(dt, d, driving_route[segment_pos.index])

        # update the vehicle
        vehicle.time_offset += d
        vehicle.set_position(segment_pos)

        if vehicle.current_node == vehicle.dest_node:
            # stop the processing in case the vehicle reached the end
            vehicle.active = False

    return vehicle


def distance_duration(driving_route, departure_time, stop_ditance, los_db): # TODO: implement los db
    # TODO: corner case what if the stop distance is longer than the entire route

    distance = 0.0
    p = SegmentPositin(0, 0.0)
    dt = departure_time
    level_of_services = []

    while distance < stop_distance:
        seg = driving_route[p.index]
        los = los_db.get(dt, seg)

        if los == float("inf"):  # stucks in traffic jam; not moving
            return (float("inf"), None, None)

        time, next_segment_pos  = driving_route.advance(p, dt, los)
        d = time - departure_time

        if p.index == next_segment_pos.index:
            # movement on the same segment
            distance += next_segment_pos.start - p.start
        else:
            # if the next segment is different, i.e. its index is higher than
            # the rest distance of the previous segment is added.
            distance += seg.length - p.start

        if distance > stop_distance:
            # round down to the stop distance

            # deacrease the segment
            dd = distance - stop_distance
            if next_segment_pos.start - dd < 0:
                segment_pos = SegmentPosition(p.index, seg.length + (next_segment.start - dd))
            else:
                segment_pos.start -= dd

            # deacrease the duration => I need to know the last assigned speed
            # TODO:

        level_of_services.append(los)
        dt += d
        p = next_segment_pos

        # duration, next segment position, average level of service
    return (dt - departure_time, p, sum(level_of_services) / len(level_of_services)) # TODO: is average ok?


def route_rank(driving_route, departure_time, gv, gv_distance: float, prob_profiles, nsamples):
    """The smaller the rank the better."""

    return 1.0

    # TODO --- uncoment when los db will be implemented

    # p = SegmentPosition(0, 0.0)
    # dt = departure_time

    # # driving by global view
    # dur_ff, _, _ = distance_duration(driving_route, dt, gv_distance, None) # TODO: exchange free flow for db
    # dur, continue_pos, avg_los = distance_duration(driving_route, dt, gv_distance, None) # TODO: exchange for global view

    # if avg_los == float("inf"):
    #     return float("inf")

    # gv_delay = dur - dur_ff
    # probable_delay = pobable_duration(driving_route, continue_pos, dt + gv_delay, prob_profiles, nsamples)

    # # NOTE: the more the avg_los will be close to zero the more the probable delay will be prolonged
    # return gv_delay + probable_delay / (1 - avg_los)


def ptdr(vehicle, departure_time, k_routes, gv, gv_distance, prob_profiles, nsamples):
    osm_routes = vehicle.k_shortest_paths(k_routes)  # TODO: unify usage of _path_ and _route_ terms
    possible_driving_routes = list(
        map(lambda osm_route: Route(osm_route_to_segments(osm_route, vehicle.routing_map),
                                    vehicle.frequency),
            osm_routes))

    # pick the driving route with the smallest deylay
    if len(possible_driving_routes) > 1:
        ranks = map(lambda driving_route: route_rank(
            driving_route, departure_time, gv, gv_distance, prob_profiles, nsamples), possible_driving_routes)
        indexed_ranks = sorted(enumerate(ranks), key=lambda indexed_rank: indexed_rank[1])

        best_route_index, _ = indexed_ranks[0]
    else:
        best_route_index = 0

    return (osm_routes[best_route_index], possible_driving_routes[best_route_index])
