import pandas as pd
import numpy as np
from dataclasses import astuple
from typing import List, Tuple
from datetime import datetime, timedelta


class GlobalView:

    TIMESTAMP = "timestamp"
    SEGMENT_ID = "segment_id"
    VEHICLE_ID = "vehicle_id"

    def __init__(self, data=None):
        if data is None:
            self.data = pd.DataFrame({
                GlobalView.TIMESTAMP: pd.Series(dtype="datetime64[s]"),
                GlobalView.SEGMENT_ID: pd.Series(dtype="str"),
                GlobalView.VEHICLE_ID: pd.Series(dtype="int64"),
            })
            self.data.set_index([GlobalView.TIMESTAMP, GlobalView.SEGMENT_ID], inplace=True)
        else:
            self.data = data

    def add(self, car_id, hrs: List[Tuple[datetime, str]]):
        midx = pd.MultiIndex.from_tuples(
            [(np.datetime64(dt, 's'), seg_id)
             for dt, seg_id in hrs],
            names=[GlobalView.TIMESTAMP, GlobalView.SEGMENT_ID])

        new_data = pd.DataFrame(
            np.full(len(midx), car_id), index=midx, columns=[GlobalView.VEHICLE_ID])

        self.data = pd.concat([self.data, new_data])
        self.data.sort_index()

    def vehicles_in_time_at_segment(self, datetime, segment_id, tolerance=None):
        dt = np.datetime64(datetime, 's')
        tolerance = np.timedelta64(tolerance if tolerance is not None else timedelta(seconds=0))

        return self.data.query(f"{GlobalView.TIMESTAMP} >= '{dt - tolerance}' and "
                               f"{GlobalView.TIMESTAMP} <= '{dt + tolerance}' and "
                               f"{GlobalView.SEGMENT_ID} == '{segment_id}'")

    def store(self, path):
        self.data.to_pickle(path)

    def load(self, path):
        pass # TODO: implement laoading maybe as param of init

    def __len__(self):
        return len(self.data)

    def raw_data(self):
        return self.data
