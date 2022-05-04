
"""A distributed traffic simulator."""

import logging
import os
import random
import functools
import time
import math
from multiprocessing.pool import ThreadPool
from multiprocessing import Pool
from datetime import datetime, timedelta
from dataclasses import dataclass
from evkit.comm import allreduce, distribute, init

from probduration import HistoryHandler

from ruth.losdb import ProbProfileDb, GlobalViewDb

from ruth.vehicle import Vehicle
from ruth.distsim import load_vehicles, save_vehicles, advance_vehicle, alternatives, ptdr
from ruth.globalview import GlobalView
from ruth.utils import round_timedelta


import pickle

# TODO: remove after tests
import pandas as pd
from os.path import exists


logger = logging.getLogger(__name__)


@dataclass
class CycleInfo:
    time: object
    updates: list


@dataclass
class StepInfo:
    step: int
    n_active: int
    time_for_alternatives: timedelta
    time_for_ptdr: timedelta
    time_for_advance: timedelta

    @staticmethod
    def from_row(step, n_active, time_for_alternatives_s, time_for_ptdr_s, time_for_advance_s):
        time_for_ptdr = None if math.isnan(time_for_ptdr_s) else timedelta(seconds=time_for_ptdr_s)
        time_for_advance = None if math.isnan(time_for_advance_s) else timedelta(seconds=time_for_advance_s)
        return StepInfo(step, n_active,
                        timedelta(seconds=time_for_alternatives_s),
                        time_for_ptdr,
                        time_for_advance)

    def __repr__(self):
        sec = timedelta(seconds=1)
        time_for_ptdr = "" if self.time_for_ptdr is None else f"{self.time_for_ptdr / sec}"
        time_for_advance = "" if self.time_for_advance is None else f"{self.time_for_advance / sec}"
        return f"{self.step};{self.n_active};{self.time_for_alternatives / sec};{time_for_ptdr};{time_for_advance}"

    def __str__(self):
        sec = timedelta(seconds=1)
        return f"StepInfo(step={self.step}, active={self.n_active})"


def is_allowed(vehicle, current_offset, freq: timedelta):
    return vehicle.active and current_offset == round_timedelta(vehicle.time_offset, freq)


def update_vehicle_cached2(vehicle_route, current_offset, freq, advance_args):
    vehicle, osm_route = vehicle_route

    leap_history = []
    if is_allowed(vehicle, current_offset, freq):
        new_vehicle = advance_vehicle(vehicle, osm_route, *advance_args)
        # swap the empty history with the filled one
        leap_history, new_vehicle.leap_history = new_vehicle.leap_history, leap_history
    else:
        new_vehicle = vehicle

    return (new_vehicle, leap_history)


def update_vehicle(vehicle_route, current_offset, freq: timedelta, advance_args):
    vehicle, osm_route = vehicle_route

    leap_history = []
    if is_allowed(vehicle, current_offset, freq):
        new_vehicle = advance_vehicle(vehicle, osm_route, *advance_args)
        # swap the empty history with the filled one
        leap_history, new_vehicle.leap_history = new_vehicle.leap_history, leap_history
    else:
        new_vehicle = vehicle

    return (new_vehicle, leap_history)


def get_updates(vehicle, leap_history):
    if not leap_history:
        return None
    return (vehicle.id, leap_history)


def ptdr_(alt, departure_time, gv_db, n_samples):
    vehicle, osm_routes = alt

    if osm_routes is None:
        return (vehicle, None)

    dt = departure_time + vehicle.time_offset
    prob_profile_db = ProbProfileDb(HistoryHandler.no_limit())
    return ptdr(vehicle, osm_routes, dt, gv_db, 200, prob_profile_db, n_samples)


CURRENT_GV_DATA = None


def ptdr_cached(alt, pickled_data):
    global CURRENT_GV_DATA
    vehicle, osm_routes = alt
    #departure_time, gv_db, n_samples = pickle.loads(pickled_data)
    assert CURRENT_GV_DATA is not None
    departure_time, gv_db, n_samples = CURRENT_GV_DATA

    if osm_routes is None:
        return (vehicle, None)

    dt = departure_time + vehicle.time_offset
    prob_profile_db = ProbProfileDb(HistoryHandler.no_limit())
    return ptdr(vehicle, osm_routes, dt, gv_db, 200, prob_profile_db, n_samples)


def ptdr2_(departure_time, gv_db, n_samples):
    pickled_data = None#pickle.dumps((departure_time, gv_db, n_samples))

    return functools.partial(ptdr_cached, pickled_data=pickled_data)


def set_global_gv(data):
    global CURRENT_GV_DATA
    CURRENT_GV_DATA = pickle.loads(data)


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


    # read the state
    f_gv = "gv_walltime.parquet"
    gv_df = None
    if exists(f_gv):
        gv_df = pd.read_parquet(f_gv, engine="fastparquet")
    gv_db = GlobalViewDb(GlobalView(data=gv_df))

    f_step_info = "step_info_walltime.csv"
    step = 0
    step_info = []
    if exists(f_step_info):
        step_info_df = pd.read_csv(f_step_info, sep=';')
        step_info_df.columns = ["step", "n_active", "time_for_alternatives_s", "time_for_ptdr_s", "time_for_advance_s"]
        step_info = [StepInfo.from_row(**row) for _, row in step_info_df.iterrows()]
        step = step_info[-1].step

    f_vehicles = "vehicles_walltime.pickle"
    if exists(f_vehicles):
        vehicles_df = pd.read_pickle(f_vehicles)
        vehicles = [Vehicle(**row.to_dict()) for (_, row) in vehicles_df.iterrows()]

    process_count = 8

    current_offset = timedelta(seconds=0)
    with Pool(processes=process_count) as p:
        sim_start = datetime.now()
        walltime = timedelta(minutes=5)
        walltime_saved = False

        while current_offset is not None:
            round_start = datetime.now()
            current_offset_ = round_timedelta(current_offset, round_freq)
            allowed_vehicles = list(filter(lambda v: is_allowed(v, current_offset_, round_freq), vehicles))

            s = datetime.now()
            # a. compute alternatives
            alts = list(filter(None, p.map(functools.partial(alternatives, k=4), allowed_vehicles)))
            # b. inactive the vehicles for which there is no alternative
            for v, osm_routes in alts:
                if osm_routes is None:
                    v.active = False
                    v.status = "no-alternative for the origin/destination"
            time_for_alternatives = datetime.now() - s

            if not alts:
                # in case of no alternativces for the allowed vehicles update their activity and collect leap history
                new_vehicles = []
                for v in allowed_vehicles:
                    v.active = False
                    leap_history = []
                    leap_history, v.leap_history = v.leap_history, leap_history
                    new_vehicles.append((v, leap_history))

                time_for_ptdr = None
                time_for_advance = None
            else:
                # in case there are alternatives compute the best routes and update the vehicles

                # s = datetime.now()
                # pickled_gv = pickle.dumps((departure_time, gv_db, n_samples))
                # p.map(set_global_gv, [pickled_gv] * process_count)
                # e = datetime.now()
                # print(f"{(e - s).total_seconds()} GV transfer")

                s = datetime.now()
                # cached_ptdr = ptdr2_(departure_time, gv_db, n_samples)
                # bests = p.map(cached_ptdr, alts)
                bests = list(map(functools.partial(ptdr_,
                                                   departure_time=departure_time,
                                                   gv_db=gv_db,
                                                   n_samples=n_samples), alts))
                # bests = thread_pool.map(functools.partial(ptdr_,
                #                                           departure_time=departure_time,
                #                                           gv_db=gv_db,
                #                                           n_samples=n_samples), alts)
                time_for_ptdr = datetime.now() - s

                s = datetime.now()
                #cached_update_vehicle = update_vehicle(current_offset_, round_freq, advance_args=(departure_time, gv_db.gv, 200, n_samples))
                #new_vehicles = p.map(cached_update_vehicle, bests)
                new_vehicles = []
                for best in bests:
                    new_vehicles.append(update_vehicle(best,
                                                       current_offset_,
                                                       round_freq,
                                                       advance_args=(
                                                           departure_time,
                                                           gv_db.gv,
                                                           200,
                                                           n_samples)))
                    # new_vehicles.append(update_vehicle_cached2(best, current_offset=current_offset_,
                    #                                     freq=round_freq,
                    #                                     advance_args=(
                    #                                         departure_time,
                    #                                         gv_db.gv,
                    #                                         200,
                    #                                         n_samples
                    #                                     )))
                #new_vehicles = thread_pool.map(functools.partial(update_vehicle,
                #                                        current_offset=current_offset_,
                #                                        freq=round_freq,
                #                                        advance_args=(
                #                                            departure_time,
                #                                            gv_db.gv,
                #                                            200,
                #                                            n_samples
                #                                        )), bests)
                time_for_advance = datetime.now() - s

            # update the vehicles list based on the new vehicles
            vd = dict((v.id, v) for v in vehicles)
            updates = []
            for i in range(len(new_vehicles)):
                v, lhu = new_vehicles[i]
                updates.append( (v.id, lhu) )
                vd[v.id], new_vehicles[i] = v, vd[v.id]

            vehicles = list(vd.values())

            current_offset = min(filter(None, map(lambda v: v.time_offset if v.active else None, vehicles)), default=None)

            # update global view
            for vehicle_id, lhu in updates:
                gv_db.gv.add(vehicle_id, lhu)

            step_info.append(StepInfo(step, len(allowed_vehicles), time_for_alternatives, time_for_ptdr, time_for_advance))
            logger.info(f"{step_info[-1]}")

            if datetime.now() - sim_start >= walltime and not walltime_saved:
                gv_db.gv.store("gv_walltime.parquet")
                save_vehicles(vehicles, "vehicles_walltime.pickle")
                with open(f"step_info_walltime.csv", "w") as f:
                    f.write("\n".join(map(repr, step_info)))

                walltime_saved = True
                break

            round_end = datetime.now()
            round_duration = round_end - round_start
            x = time_for_alternatives.total_seconds()
            y = time_for_ptdr.total_seconds() if time_for_ptdr is not None else 0
            z = time_for_advance.total_seconds() if time_for_advance is not None else 0
            logger.info(f"alt: {x}, ptdr: {y}, advance: {z}")
            #logger.info(f"gv: {round_duration.total_seconds() - (x + y + z)}, alt + ptdr + advance: {x + y + z}, total: {round_duration.total_seconds()}")

            step += 1

    sim_end = datetime.now()
    length = sim_end - sim_start
    print(f"Duration: {length}")

    gv_history.store("gv_end.pickle")
    save_vehicles(vehicles, "vehicles_end.pickle")
    with open(f"step_info_end.csv", "w") as f:
        f.write("\n".join(map(repr, step_info)))
    return (gv_history, step_info)


def simulate(input_path: str, *args, **kwargs):
    """Distributed traffic simulator."""
    init()
    cars = load_vehicles(input_path)
    # return distribute(cars, main_cycle, args=args, kwargs=kwargs)
    return main_cycle(cars, *args, **kwargs)
