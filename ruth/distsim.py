"""A distributed traffic simulator."""

import logging
import os
import random
import pandas as pd
import dask.bag as db
from copy import copy
from dask.distributed import Client
from dataclasses import asdict

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
                return advance_vehicle(vehicle, n_samples, k_routes, departure_time, gv)
            else:
                return vehicle

        vehicles = vehicles.map(advance, dist_gv).persist()

        if intermediate_results is not None and step % (checkpoint_period * gv_update_periodl) == 0:
            # TODO: Save vehicles
            pass

        # process the leap history
        # def process_leap_history(acc, vehicle):
        #     leap_history_update = acc[:]

        #     leap_history_update.append((vehicle.id, vehicle.leap_history[:]))
        #     vehicle.leap_history.clear()

        #     return leap_history_update

        # def join_leap_histories(acc, lh_update):
        #     leap_history = acc[:]
        #     leap_history += lh_update

        #     return leap_history

        # # list of (vehicle id, leap_history)
        # leap_history_update = vehicles.fold( # TODO: why the folding does not work and nondeterministically place instead of vehilce the string: "("
        #     process_leap_history,
        #     join_leap_histories,
        #     initial=[]
        # ).compute()
        # << Fold variant of processing the leap history; which does not work properly; it crashes non-deterministically

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

    return vehicles


def advance_vehicle(vehicle, samples, k_routes, departure_time, gv):
    """Advance a vehicle on a route."""

    # compute the k shortest paths and compose driving rotes from them
    osm_routes = vehicle.k_shortest_paths(k_routes)  # TODO: unify using of _path_ and _route_ terms
    possible_driving_routes = list(
        map(lambda osm_route: Route(osm_route_to_segments(osm_route, vehicle.routing_map),
                                    vehicle.frequency),
            osm_routes))

    dt = departure_time + vehicle.time_offset
    history = HistoryHandler.no_limit()  # TODO: history get here or take as an argument?

    # pick the driving route with the smallest deylay
    if len(possible_driving_routes) > 1:
        start_pos = SegmentPosition(0, 0.0)  # TODO: the start will be given by a prefix for which was used the global view
                                             #       + departure time has corresponds
        delays = map(lambda driving_route: probable_duration(
            driving_route, start_pos, dt, history, samples), possible_driving_routes)
        indexed_delays = sorted(enumerate(delays), key=lambda indexed_delay: indexed_delay[1])

        best_route_index, _ = indexed_delays[0]
    else:
        best_route_index = 0

    # update the current route
    vehicle.set_current_route(osm_routes[best_route_index])

    # advance the vehicle on the driving route
    driving_route = possible_driving_routes[best_route_index]
    time, segment_pos = driving_route.advance(
        vehicle.segment_position, dt, 1.0)  # TODO: here will be used LoS only from global view!
    d = time - dt

    # NOTE: _assumtion_: the car stays on a single segment within one call of the `advance`
    #       method on the driving route

    if segment_pos.index < len(driving_route):  # NOTE: the segment position index may end out of segments
        vehicle.store_fcd(dt, d, driving_route[segment_pos.index])

    # update the vehicle
    vehicle.time_offset += d
    vehicle.set_position(segment_pos)

    if vehicle.current_node == vehicle.dest_node:
        # stop the processing in case the ca    r reached the end
        vehicle.active = False

    return vehicle
