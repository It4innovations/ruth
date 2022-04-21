import pandas as pd
import numpy as np
from dataclasses import astuple
from typing import List, Tuple
from datetime import datetime, timedelta


class GlobalView:

    TIMESTAMP = "timestamp"
    SEGMENT_ID = "segment_id"
    VEHICLE_ID = "vehicle_id"
    START_OFFSET = "start_offset_m"
    SPEED = "speed_mps"

    def __init__(self, data=None):
        if data is None:
            self.data = pd.DataFrame({
                GlobalView.TIMESTAMP: pd.Series(dtype="datetime64[s]"),
                GlobalView.SEGMENT_ID: pd.Series(dtype="str"),
                GlobalView.VEHICLE_ID: pd.Series(dtype="int64"),
                GlobalView.START_OFFSET: pd.Series(dtype="float64"),
                GlobalView.SPEED: pd.Series(dtype="float64")
            })
            self.data.set_index([GlobalView.TIMESTAMP, GlobalView.SEGMENT_ID], inplace=True)
        else:
            self.data = data

    def add(self, car_id, hrs: List[Tuple[datetime, str, float, float]]):
        midx = pd.MultiIndex.from_tuples(
            [(np.datetime64(dt, 's'), seg_id)
             for dt, seg_id, _, _ in hrs],
            names=[GlobalView.TIMESTAMP, GlobalView.SEGMENT_ID])

        columns = [(car_id, start_offset, speed) for _, _, start_offset, speed in hrs]
        new_data = pd.DataFrame(
            columns, index=midx, columns=[GlobalView.VEHICLE_ID, GlobalView.START_OFFSET, GlobalView.SPEED])

        self.data = pd.concat([self.data, new_data])
        self.data.sort_index()

    def vehicles_in_time_at_segment(self, datetime, segment_id, tolerance=None):
        dt = np.datetime64(datetime, 's')
        tolerance = np.timedelta64(tolerance if tolerance is not None else timedelta(seconds=0))

        return self.data.query(f"{GlobalView.TIMESTAMP} >= '{dt - tolerance}' and "
                               f"{GlobalView.TIMESTAMP} <= '{dt + tolerance}' and "
                               f"{GlobalView.SEGMENT_ID} == '{segment_id}'")

    def number_of_vehicles_in_time_at_segment(self, datetime, segment_id, tolerance=None):
        vehicles = self.vehicles_in_time_at_segment(datetime, segment_id, tolerance)

        return len(set(vehicles[GlobalView.VEHICLE_ID]))

    def level_of_service_in_time_at_segment(self, datetime, segment, tolerance=None):
        mile = 1609.344 # meters
        # density of vehicles per mile with ranges of level of service
        # https://transportgeography.org/contents/methods/transport-technical-economic-performance-indicators/levels-of-service-road-transportation/
        ranges = [
            ( (0, 12), (0.0, 0.2)),
            ((12, 20), (0.2, 0.4)),
            ((20, 30), (0.4, 0.6)),
            ((30, 42), (0.6, 0.8)),
            ((42, 67), (0.8, 1.0))]

        n_vehicles = self.number_of_vehicles_in_time_at_segment(datetime, segment.id, tolerance)

        # rescale density
        n_vehicles_per_mile = n_vehicles * mile / segment.length

        los = float("inf") # in case the vehicles are stuck in traffic jam
        for (low, high), (m, n) in ranges:
            if n_vehicles_per_mile < high:
                d = high - low  # size of range between two densities
                los = m + ((n_vehicles_per_mile - low) * 0.2 / d) # -low => shrink to the size of the range
                break

        # reverse the level of service 1.0 means 100% LoS, but the input table defines it in reverse
        return los if los == float("inf") else 1.0 - los

    def store(self, path):
        self.data.to_parquet(path, engine="fastparquet")

    def load(self, path):
        pass # TODO: implement laoading maybe as param of init

    def __len__(self):
        return len(self.data)

    def raw_data(self):
        return self.data

    def copy(self):
        return GlobalView(self.data.copy())
