
"""A distributed traffic simulator."""

import logging
import os
import random
from datetime import timedelta
from dataclasses import dataclass
from evkit.comm import allreduce, distribute, init

from ruth.distsim import load_vehicles, advance_vehicle
from ruth.globalview import GlobalView
from ruth.utils import round_timedelta


logger = logging.getLogger(__name__)


@dataclass
class CycleInfo:
    time: object
    updates: list

def update_vehicle(vehicle, current_offset, freq: timedelta, *advance_args):
    leap_history = []
    if vehicle.active and current_offset == round_timedelta(vehicle.time_offset, freq):
        new_vehicle = advance_vehicle(vehicle, *advance_args)
        # swap the empty history with the filled one
        leap_history, new_vehicle.leap_history = new_vehicle.leap_history, leap_history
    else:
        new_vehicle = vehicle

    return (new_vehicle, leap_history)


def get_updates(vehicle, leap_history):
    if not leap_history:
        return None
    return (vehicle.id, leap_history)


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

    round_freq = timedelta(seconds=gv_update_period)

    gv = GlobalView()
    step = 0
    current_offset = timedelta(seconds=0)
    while current_offset is not None:
        current_offset_ = round_timedelta(current_offset, round_freq)
        allowed_vehicles = len(list(filter(lambda v: v.active and current_offset_ == round_timedelta(v.time_offset, round_freq), vehicles)))
        logger.info(f"Starting step {step} with {allowed_vehicles} vehicles")

        gv_near_future = 200 # 200m look ahead; TODO: make a paramter
        vehicles, leap_histories = zip(*map(lambda v: update_vehicle(v,
                                                                     current_offset_,
                                                                     round_freq ,
                                                                     departure_time,
                                                                     k_routes,
                                                                     gv,
                                                                     gv_near_future,
                                                                     n_samples), vehicles))

        local_min_offset = min(filter(None, map(lambda v: v.time_offset
                                                if v.active else None, vehicles)), default=None)
        local_updates = list(filter(None, map(lambda v_lhu: get_updates(*v_lhu),
                                              zip(vehicles, leap_histories))))

        # update global view
        all_updates = allreduce([CycleInfo(local_min_offset, local_updates)])
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
