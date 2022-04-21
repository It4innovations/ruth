
import logging
import sys
import pandas as pd
import random
from datetime import timedelta

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
    current_offset = timedelta(seconds=0)
    while current_offset is not None:
        logger.info("Starting step %s", step)

        car_updates = []
        new_time = None
        new_vehicles = []
        for vehicle in vehicles:
            if current_offset == vehicle.time_offset:
                gv_near_future = 200 # 200m look ahead; make a parameter
                new_vehicle = advance_vehicle(vehicle, departure_time, k_routes, gv, gv_near_future, n_samples)
                car_updates.append((new_vehicle.id, new_vehicle.leap_history))
            else:
                new_vehicle = vehicle

            if new_vehicle.active:
                if new_time is None:
                    new_time = new_vehicle.time_offset
                else:
                    new_time = min(new_time, new_vehicle.time_offset)
            new_vehicles.append(new_vehicle)

        vehicles = new_vehicles

        # update global view
        all_updates = allreduce([CycleInfo(new_time, car_updates)])
        current_offset = min((up.time for up in all_updates if up.time is not None), default=None)

        for up in all_updates:
            for vehicle_id, lhu in up.updates:
                gv.add(vehicle_id, lhu)

        step += 1

    return gv


def simulate(input_path: str, *args, **kwargs):
    """Distributed traffic simulator."""
    init()
    cars = load_vehicles(input_path)
    return distribute(cars, main_cycle, args=args, kwargs=kwargs)
