
import logging
import sys
import pandas as pd
import random

from ruth.distsim import load_vehicles, advance_vehicle
from ruth.vehicle import Vehicle
from probduration import HistoryHandler, Route, probable_duration
from ruth.utils import osm_route_to_segments


logger = logging.getLogger(__name__)


"""A distributed traffic simulator."""

import logging
import os
import random
import pandas as pd
import evkit.dask.bag as db
from copy import copy
from dask.distributed import Client
from dataclasses import asdict

from probduration import HistoryHandler, Route, SegmentPosition, probable_duration

from ruth.utils import osm_route_to_segments
from ruth.vehicle import Vehicle
from ruth.globalview import GlobalView


logger = logging.getLogger(__name__)


def simulate(input_path: str,
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
        intermediate_results = os.path.abspath(intermediate_results)

        if not os.path.exists(intermediate_results):
            os.mkdir(intermediate_results)

    gv = GlobalView()

    vehicles = db.from_sequence(load_vehicles(input_path))

    step = 0
    while True:
        logger.info("Starting step %s", step)

        min_offset = vehicles.reduction(compute_min, min).compute()

        if min_offset == float("inf"):
            # No active cars
            break

        vehicles = vehicles.map(advance, gv, min_offset, n_samples, k_routes, departure_time).persist()

        if intermediate_results is not None and step % (checkpoint_period * gv_update_period) == 0:
            # TODO: Save vehicles
            pass

        leap_history_update = vehicles.reduction(
            process_leap_history,
            join_leap_histories
        ).compute()

        if leap_history_update: # update only if there is data
            for vehicle_id, lhu in leap_history_update:
                gv.add(vehicle_id, lhu)

        step += 1

    return (gv, vehicles)


def compute_min(partitions):
    return min((v.time_offset for v in partitions if v.active), default=float("inf"))


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


def advance(vehicle, gv, min_offset, n_samples, k_routes, departure_time):
    data = asdict(vehicle)
    data["leap_history"] = []
    vehicle = Vehicle(**data)

    if vehicle.time_offset == min_offset:
        return advance_vehicle(vehicle, n_samples, k_routes, departure_time, gv)
    else:
        return vehicle
