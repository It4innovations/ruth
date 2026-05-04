import logging
import os
import pickle
import glob
from dataclasses import InitVar, dataclass
from datetime import datetime, timedelta
from random import random, seed as rnd_seed
from typing import Dict, List, Optional

import h5py
import pandas as pd

from .queues import QueuesManager
from ..data.map import BBox, Map
from ..data.segment import LengthMeters, Segment, SpeedMps
from ..fcd_history import FCDHistory
from ..vehicle import Vehicle

try:
    from ..globalview_wrapper import GlobalView
    logging.info("Using C++ GlobalView module.")
except ImportError:
    logging.warning("C++ GlobalView module not found, using Python fallback.")
    from ..globalview import GlobalView


@dataclass(frozen=True)
class FCDRecord:
    datetime: datetime
    vehicle_id: int
    segment: Segment
    offset_from_start: LengthMeters
    vehicle_speed_mps: SpeedMps
    status: str
    active: bool


@dataclass
class StepInfo:
    simulation_offset: timedelta
    step: int
    n_active: int
    duration: timedelta
    parts: Dict[str, float]
    need_new_route: int


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
    map_update_freq_s: timedelta = timedelta(seconds=1)
    los_vehicles_tolerance: timedelta = timedelta(seconds=0)
    travel_time_limit_perc: float = 0.0
    seed: InitVar[Optional[int]] = None
    speeds_path: str = None
    buffer_size: int = 10_000
    max_records_per_file: int = int(1e9)
    stuck_detection: int = 0
    plateau_default_route: bool = False
    fcd_history_base_name: str = "fcd_history"

    def __post_init__(self, seed: Optional[int]):
        if seed is not None:
            rnd_seed(seed)
        self.rnd_gen = random

        if self.fcd_history_base_name:
            # when starting a new simulation, there must be no existing part files
            base_no_ext = os.path.splitext(self.fcd_history_base_name)[0]
            existing_parts = glob.glob(f"{base_no_ext}-part*.h5")
            if existing_parts:
                raise FileExistsError(f"FCD history base '{self.fcd_history_base_name}' conflicts with existing files: {existing_parts}")

class Simulation:
    """A simulation state."""

    def __init__(self, vehicles: List[Vehicle], setting: SimSetting, bbox: BBox, map_download_date: str):
        """
        Construct a new simulation.
        """

        self.setting = setting
        self.history = FCDHistory(self.setting.fcd_history_base_name, self.setting.buffer_size, self.setting.max_records_per_file)
        self.bbox = bbox
        self.map_download_date = map_download_date
        self._routing_map = Map(self.bbox, download_date=self.map_download_date, with_speeds=True)
        self.global_view = GlobalView(routing_map=self._routing_map)
        self.vehicles = vehicles
        self.steps_info = []
        self.duration = timedelta(seconds=0)
        self.queues_manager = QueuesManager()
        self.last_saved_speeds = None
        self._freq_seconds: int = int(self.setting.round_freq.total_seconds())

    def __getstate__(self):
        self.last_saved_speeds = {}
        for node_from, node_to, data in self.routing_map.current_network.edges(data=True):
            segment_id = (node_from, node_to)
            self.last_saved_speeds[segment_id] = data.get('current_speed', None)

        d = self.__dict__.copy()
        if "_routing_map" in d:
            d.pop("_routing_map")
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        if "last_saved_speeds" not in d:
            self.last_saved_speeds = None
        if "_freq_seconds" not in d:
            self._freq_seconds = int(self.setting.round_freq.total_seconds())
        self._routing_map = None  # lazy init

        # TODO: check when necessary to recreate global_view and when not
        if "global_view" not in d:
            self.global_view = GlobalView(routing_map=self.routing_map)

        # if it is cpp globalview, set the routing map there as well
        if hasattr(self.global_view, 'set_routing_map'):
            self.global_view.set_routing_map(self.routing_map)

    @property
    def routing_map(self):
        if self._routing_map is None:
            self._routing_map = Map(self.bbox, download_date=self.map_download_date, with_speeds=True)
        return self._routing_map

    @property
    def random(self):
        return self.setting.rnd_gen()

    def is_vehicle_within_offset(self, vehicle: Vehicle, offset_seconds: int):
        """Check if vehicle should move at this offset. offset_seconds must be pre-rounded."""
        if not vehicle.active:
            return False

        freq = self._freq_seconds
        vehicle_rounded = int(freq * round(vehicle.time_offset.total_seconds() / freq))
        return vehicle_rounded == offset_seconds

    def round_time_offset(self, offset):
        """Round offset to nearest frequency. Returns (timedelta, rounded_seconds)."""
        td_seconds = offset.total_seconds()
        freq = self._freq_seconds
        rounded_seconds = freq * round(td_seconds / freq)
        return timedelta(seconds=rounded_seconds), rounded_seconds

    def compute_current_offset(self):
        return min(filter(None, map(lambda v: v.time_offset if v.active else None, self.vehicles)),
                   default=None)

    def update(self, fcds: List[FCDRecord]):
        # TODO: consider adding a batch update to python global view as well for better performance
        if hasattr(self.global_view, 'add_batch'):
            self.global_view.add_batch(fcds)
        else:
            for fcd in fcds:
                self.global_view.add(fcd)

    def drop_old_records(self, offset_threshold):
        if offset_threshold is not None:
            self.global_view.drop_old(self.setting.departure_time + offset_threshold)

    def save_step_info(self, simulation_offset, step, n_active, duration, parts, need_new_route):
        self.steps_info.append(StepInfo(simulation_offset, step, n_active, duration, parts, need_new_route))

    def steps_info_to_dataframe(self):
        if not self.steps_info:
            return pd.DataFrame()

        first = self.steps_info[0]
        return pd.DataFrame(
            [(si.simulation_offset, si.step, si.n_active, si.duration / timedelta(milliseconds=1), *si.parts.values())
             for si in self.steps_info],
            columns=["simulation_offset", "step", "n_active", "duration"] + list(first.parts.keys()))

    def get_length(self):
        """
        This function will be deprecated soon with migration to h5 storage for FCD history.
        """
        logging.warning("This function will be deprecated soon with migration to h5 storage for FCD history.")
        if not self.history.fcd_history:
            raise ValueError("No FCD history available.")
        return self.history.fcd_history[-1].datetime - self.history.fcd_history[0].datetime

    def get_vehicle_ids_not_finished(self):
        return set([v.id for v in self.vehicles if v.active])

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

    @staticmethod
    def load_h5_df(path):
        with h5py.File(path, 'r') as f:
            df = pd.DataFrame(f['fcd'][:])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

            return {
                "df": df,
                "departure_time": f.attrs['departure_time'],
                "bbox": BBox(*f.attrs['bbox']),
                "download_date": f.attrs['download_date'],
                "computational_time": f.attrs['computational_time']
            }
