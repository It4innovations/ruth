import pandas as pd
import numpy as np
import operator
from dataclasses import astuple
from typing import List, Tuple
from datetime import datetime, timedelta
import pickle
import random


class GlobalView:

    def __init__(self, data=None):
        self.data = [] if data is None else data

    def add(self, vehicle_id, hrs):
        rows = [(dt, seg_id, vehicle_id, start_offset, speed, status)
               for dt, seg_id, start_offset, speed, status in hrs]
        self.data += rows

    def vehicles_in_time_at_segment(self, datetime, segment_id, tolerance=None):
        tolerance = tolerance if tolerance is not None else timedelta(seconds=0)

        return [(dt, seg_id, *rest) for dt, seg_id, *rest in self.data
                if dt >= datetime - tolerance and dt <= datetime + tolerance and seg_id == segment_id]

    def number_of_vehicles_in_time_at_segment(self, datetime, segment_id, tolerance=None):
        rows = self.vehicles_in_time_at_segment(datetime, segment_id, tolerance)
        vehicles = filter(lambda row: row[2], rows)

        return len(set(vehicles))

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
        # sort the data
        self.data.sort(key=operator.itemgetter(0,1))

        with open(path, 'wb') as f:
            pickle.dump(self.data, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            return GlobalView(data=data)

    def __len__(self):
        return len(self.data)

    def drop_old(self, dt_threshold):
        self.data.sort(key=lambda row: row[0])

        for i, row in enumerate(self.data):
            if row[0] >= dt_threshold:
                self.data = self.data[i:]
                break
