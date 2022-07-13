"""A distributed traffic simulator."""

import logging
import os
import random
from typing import final
import pandas as pd
import dask.bag as db
from copy import copy
from itertools import groupby
from dask.distributed import Client
from dataclasses import asdict
from datetime import timedelta
from networkx.exception import NetworkXNoPath
import time

from probduration import HistoryHandler, Route, SegmentPosition, probable_delay, avg_delays, VehiclePlan

from ruth.utils import osm_route_to_segments, timer
from ruth.vehicle import Vehicle
from ruth.globalview import GlobalView
from ruth.losdb import ProbProfileDb, GlobalViewDb, FreeFlowDb


logger = logging.getLogger(__name__)


def load_vehicles(input_path: str):
    logger.info("Loading data ... %s", input_path)
    df = pd.read_parquet(input_path, engine="fastparquet")
    return [Vehicle(**row.to_dict()) for (_, row) in df.iterrows()]


def save_vehicles(vehicles, output_path: str):
    logger.info("Saving vehicles ... %s", output_path)

    df = pd.DataFrame([asdict(v) for v in vehicles])
    df.to_pickle(output_path)


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

        leap_history_update = vehicles.reduction(
            process_leap_history,
            join_leap_histories
        ).compute()

        if leap_history_update: # update only if there is data
            for vehicle_id, lhu in leap_history_update:
                gv.add(vehicle_id, lhu)

            dist_gv = c.scatter(gv, broadcast=True)

        def clear_leap_history(vehicle):
            data = asdict(vehicle)
            data["leap_history"] = []
            return Vehicle(**data)

        vehicles = vehicles.map(clear_leap_history).persist()

        step += 1

    return (gv, vehicles)


def advance_vehicle(vehicle_, osm_route, departure_time, gv, gv_distance, nsamples=1):
    """Advance a vehicle on a route."""

    vehicle = Vehicle(**asdict(vehicle_))  # make a copy of vehicle as the functions should be stateless

    dt = departure_time + vehicle.time_offset

    # LoS databases
    prob_profile_db = ProbProfileDb(HistoryHandler.no_limit())  # TODO: history get here or take as an argument?
    gv_db = GlobalViewDb(gv)

    # advance the vehicle on the driving route
    # best_route = ptdr(vehicle, dt, k_routes, gv_db, gv_distance, prob_profile_db, nsamples)
    # if best_route is None:
    #     vehicle.active = False
    #     return vehicle


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
        time = dt + vehicle.frequency
        vehicle.time_offset += vehicle.frequency
    else:
        time, segment_pos, assigned_speed_mps = driving_route.advance(
            vehicle.segment_position, dt, los)
        d = time - dt

        # NOTE: _assumtion_: the car stays on a single segment within one call of the `advance`
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


def distance_duration(driving_route, departure_time, stop_distance, los_db):
    # TODO: corner case what if the stop distance is longer than the entire route

    distance = 0.0
    p = SegmentPosition(0, 0.0)
    dt = departure_time
    level_of_services = []

    while distance < stop_distance:
        if p.index >= len(driving_route):
            break

        seg = driving_route[p.index]
        los = los_db.get(dt, seg, random.random())

        if los == float("inf"):  # stucks in traffic jam; not moving
            return (float("inf"), None, None)

        time, next_segment_pos, assigned_speed_mps  = driving_route.advance(p, dt, los)
        d = time - dt

        if p.index == next_segment_pos.index:
            # movement on the same segment
            distance += next_segment_pos.start - p.start
        else:
            # if the next segment is different, i.e. its index is higher than
            # the rest distance of the previous segment is added.
            distance += seg.length - p.start

        if distance > stop_distance:
            # round down to the stop distance

            # deacrease the distance
            dd = distance - stop_distance
            if next_segment_pos.start - dd < 0:
                next_segment_pos = SegmentPosition(p.index, seg.length + (next_segment_pos.start - dd))
            else:
                next_segment_pos.start -= dd

            # deacrease the duration by overpass time
            over_duration = dd / assigned_speed_mps
            d -= timedelta(seconds=over_duration)

        level_of_services.append(los)
        dt += d
        p = next_segment_pos

        # duration, next segment position, average level of service

    if len(level_of_services) == 0:
        return (float("inf"), None, None)

    avg_los = sum(level_of_services) / len(level_of_services)
    return (dt - departure_time, p, avg_los) # TODO: is average ok?


def route_rank(driving_route, departure_time, gv_db, gv_distance: float, prob_profile_db, nsamples):
    """The smaller the rank the better."""

    # driving by global view
    with timer("distance_duration"):
        dur_ff, _, _ = distance_duration(driving_route, departure_time, gv_distance, FreeFlowDb())
        dur, continue_pos, avg_los = distance_duration(driving_route, departure_time, gv_distance, gv_db)

    if dur == float("inf"):
        return timedelta.max

    gv_delay = dur - dur_ff
    with timer("probable_delay"):
        prob_delay = probable_delay(driving_route, continue_pos, departure_time + gv_delay, prob_profile_db.prob_profiles, nsamples)

    if avg_los < 1.0:
        # NOTE: the more the avg_los will be close to zero the more the probable delay will be prolonged
        return gv_delay + prob_delay / (1.0 - avg_los)
    return gv_delay + prob_delay


def alternatives(vehicle, k):
    try:
        osm_routes = vehicle.k_shortest_paths(k)  # TODO: unify usage of _path_ and _route_ terms
        if osm_routes is None:
            return (vehicle, None)
    except (NetworkXNoPath):
        return None

    return (vehicle, osm_routes)



def ptdr(vehicle, osm_routes, departure_time, gv_db, gv_distance, prob_profile_db, nsamples):
    with timer("possible_driving_routes"):
        possible_driving_routes = list(
            map(lambda osm_route: Route(osm_route_to_segments(osm_route, vehicle.routing_map),
                                        vehicle.frequency),
                osm_routes))

    # pick the driving route with the smallest deylay
    if len(possible_driving_routes) > 1:
        ranks = map(lambda driving_route: route_rank(
            driving_route, departure_time, gv_db, gv_distance, prob_profile_db, nsamples), possible_driving_routes)
        indexed_ranks = sorted(enumerate(ranks), key=lambda indexed_rank: indexed_rank[1])

        best_route_index, _ = indexed_ranks[0]
    else:
        best_route_index = 0

    return (vehicle, osm_routes[best_route_index])


def gv_shifts_(plans, gv_db, gv_distance, pool=None):
    fmap = map
    if pool is not None:
        fmap = pool.map

    ff_db = FreeFlowDb()
    def dd(plan):
        dur_ff, _, _ = distance_duration(plan.route, plan.departure_time, gv_distance, ff_db)
        duration, position, los = distance_duration(plan.route, plan.departure_time, gv_distance, gv_db)

        gv_delay = duration - dur_ff
        return (VehiclePlan(plan.id, plan.route, position, plan.departure_time + duration), los, gv_delay)

    return fmap(dd, plans)


def ptdrs_(shifted_plans, prob_profiles, nsamples):

    plans, loses, gv_delays = zip(*shifted_plans)

    # heavy function performed by ryon in rust
    prob_delays = avg_delays(list(plans), prob_profiles, nsamples)

    def by_plan_id(data):
        plan, *_ = data
        return plan.id

    def by_ranks(data):
        _, los, gv_delay, prob_delay = data
        if los < 1.0:
            # NOTE: the more the avg_los will be close to zero the more the probable delay will be prolonged
            return gv_delay + prob_delay / (1.0 - los)
        return gv_delay + prob_delay

    data = groupby(sorted(zip(plans, loses, gv_delays, prob_delays), key=by_plan_id), by_plan_id)

    def select_best(data):
        plan_id, group = data
        # returh the best plan
        return sorted(list(group), key=by_ranks)[0][0]

    bests = list(map(select_best, data))

    return bests
