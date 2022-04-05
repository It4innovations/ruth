
import logging
import sys
import pandas as pd
import random

from evkit.comm import allreduce, distribute, init

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
from dataclasses import asdict, dataclass

from probduration import HistoryHandler, Route, SegmentPosition, probable_duration

from ruth.utils import osm_route_to_segments
from ruth.vehicle import Vehicle
from ruth.globalview import GlobalView


logger = logging.getLogger(__name__)


@dataclass
class CycleInfo:
    time: object
    updates: list


def main_cycle(vehicles,
               departure_time,
             k_routes,
             n_samples,
             seed,
             gv_update_period,
             intermediate_results,
             checkpoint_period):
    if intermediate_results is not None:
        intermediate_results = os.path.abspath(intermediate_results)

        if not os.path.exists(intermediate_results):
            os.mkdir(intermediate_results)

    if seed is not None:
        random.seed(seed)  # used for tests: 660277

    gv = GlobalView()
    step = 0
    current_offset = departure_time
    while current_offset:
        logger.info("Starting step %s", step)

        car_updates = []
        new_time = None
        for vehicle in vehicles:
            if current_offset == vehicle.time_offset:
                advance_vehicle(vehicle, n_samples, k_routes, departure_time, gv)
                car_updates.append((vehicle.id, vehicle.leap_history))
            if vehicle.active:
                if new_time is None:
                    new_time = vehicle.time_offset
                else:
                    new_time = min(new_time, vehicle.time_offset)

        all_updates = allreduce([CycleInfo(new_time, car_updates)])
        current_offset = min((up.time for up in all_updates), default=None)

        for up in all_updates:
            for vehicle_id, lhu in up.updates:
                gv.add(vehicle_id, lhu)

        step += 1


def simulate(input_path: str, *args, **kwargs):
    """Distributed traffic simulator."""
    init()
    cars = load_vehicles(input_path)
    return distribute(cars, main_cycle, args=args, kwargs=kwargs)
#
# def compute_min(partitions):
#     return min((v.time_offset for v in partitions if v.active), default=float("inf"))

#
# def process_leap_history(partitions):
#     leap_history_update = []
#     for vehicle in partitions:
#         if vehicle.leap_history:
#             # process only non-empty history
#             leap_history_update.append((vehicle.id, vehicle.leap_history[:]))
#     return leap_history_update
#
#
# def join_leap_histories(lh_updates):
#     leap_history = []
#     for lh_update in lh_updates:
#         leap_history += lh_update
#     return leap_history
#
#
# # def advance(vehicle, gv, min_offset, n_samples, k_routes, departure_time):
# #     data = asdict(vehicle)
# #     data["leap_history"] = []
# #     vehicle = Vehicle(**data)
#
