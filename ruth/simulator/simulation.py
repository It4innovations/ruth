import pickle

from dataclasses import dataclass, field, InitVar
from datetime import datetime, timedelta
from random import random, seed as rnd_seed
from typing import List, Tuple, Optional

from ..globalview import GlobalView
from ..losdb import GlobalViewDb
from ..vehicle import Vehicle
from ..utils import round_timedelta


@dataclass
class VehicleUpdate:
    """Join the updated vehicle (after advance) and leap history of that move."""

    vehicle: Vehicle
    leap_history: List[Tuple[datetime, str, float, float]]  # TODO: make a leap history data class


@dataclass
class StepInfo:
    step: int
    n_active: int
    duration: timedelta


@dataclass
class SimSetting:
    """A simulation setting.

    Args:
    -----
        departure_time: datetime
          Departure time of the simulation.
        round_freq: timedelta
          Rounding frequency for picking active vehicles to move. The bigger the frequency is the more vehicles
          is marked as active during one step; but the simulation is less precise. On the other hand, the frequency
          of 1s cases that only a few vehicles will be allowed and the computation will be almost sequential.
        seed : Optional[int] = None
          In case the fixed random generator; if `None` random generator is initialized with a timestamp.

    Attributes:
    -----------
        rnd_gen: random
          A random generator providing numbers between [0.0, 1.0).

    """
    departure_time: datetime
    round_freq: timedelta
    k_alternatives: int = 1
    seed: InitVar = None

    def __post_init__(self, seed):
        if seed is not None:
            rnd_seed(seed)
        self.rnd_gen = random


class Simulation:
    """A simulation state."""

    def __init__(self, vehicles, setting: SimSetting):
        """
        Construct a new simulation.

        Parameters:
        -----------
            vehicles: List[Vehicles]
              a list of vehicles for simulation
            setting: SimSetting
        """

        self.history = GlobalView()  # history record
        self.global_view = GlobalView()  # active global view
        self.vehicles = vehicles
        self.setting = setting
        self.steps_info = []
        self.duration = timedelta(seconds=0)

    @property
    def random(self):
        return self.setting.rnd_gen()

    @property
    def global_view_db(self):
        return GlobalViewDb(self.global_view)

    def is_vehicle_within_offset(self, vehicle: Vehicle, offset):
        return vehicle.active and offset == self.round_time_offset(vehicle.time_offset)

    def round_time_offset(self, offset):
        return round_timedelta(offset, self.setting.round_freq)

    def compute_current_offset(self):
        return min(filter(None, map(lambda v: v.time_offset if v.active else None, self.vehicles)), default=None)

    def update(self, updates: List[VehicleUpdate]):
        vd = dict((v.id, v) for v in self.vehicles)
        for update in updates:
            # store advanced vehicle
            vd[update.vehicle.id] = update.vehicle
            # update global view
            self.global_view.add(update.vehicle.id, update.leap_history)
            self.history.add(update.vehicle.id, update.leap_history)

        self.vehicles = list(vd.values())

    def drop_old_records(self, offset_threshold):
        if offset_threshold is not None:
            self.global_view.drop_old(self.setting.departure_time + offset_threshold)

    def save_step_info(self, step, n_active, duration):
        self.steps_info.append(StepInfo(step, n_active, duration))

    @property
    def last_step(self):
        return self.steps_info[-1]

    @property
    def number_of_steps(self):
        return len(self.steps_info)

    def finished(self):
        return all(not v.active for v in self.vehicles)

    def store(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
