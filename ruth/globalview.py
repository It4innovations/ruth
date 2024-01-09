from collections import defaultdict
from datetime import timedelta
from typing import Dict, List, TYPE_CHECKING

from .data.segment import SegmentId, SpeedKph

if TYPE_CHECKING:
    from .simulator.simulation import FCDRecord


class GlobalView:
    def __init__(self):
        self.fcd_by_segment: Dict[SegmentId, List[FCDRecord]] = defaultdict(list)

    def add(self, fcd: "FCDRecord"):
        self.fcd_by_segment[fcd.segment.id].append(fcd)

    def number_of_vehicles_ahead(self, datetime, segment_id, tolerance=None, vehicle_id=-1,
                                 vehicle_offset_m=0):
        # counts all vehicles ahead of the vehicle with vehicle_id at the given segment_id at given time range
        # if case vehicle_id not set, then all vehicles are counted

        tolerance = tolerance if tolerance is not None else timedelta(seconds=0)
        vehicles = set()
        for fcd in self.fcd_by_segment.get(segment_id, []):
            if datetime - tolerance <= fcd.datetime <= datetime + tolerance:
                if fcd.vehicle_id != vehicle_id and fcd.start_offset > vehicle_offset_m:
                    vehicles.add(fcd.vehicle_id)
        return len(vehicles)

    def level_of_service_in_front_of_vehicle(self, datetime, segment, vehicle_id=-1,
                                             vehicle_offset_m=0, tolerance=None):
        mile = 1609.344  # meters
        # density of vehicles per mile with ranges of level of service
        # https://transportgeography.org/contents/methods/transport-technical-economic-performance-indicators/levels-of-service-road-transportation/
        ranges = [
            ((0, 12), (0.0, 0.2)),
            ((12, 20), (0.2, 0.4)),
            ((20, 30), (0.4, 0.6)),
            ((30, 42), (0.6, 0.8)),
            ((42, 67), (0.8, 1.0))]

        n_vehicles = self.number_of_vehicles_ahead(datetime, segment.id, tolerance,
                                                   vehicle_id, vehicle_offset_m)

        # NOTE: the ending length is set to avoid massive LoS increase at the end of the segments and also on short
        # segments, can be replaced with different LoS ranges for different road types in the future
        ending_length = 200
        rest_segment_length = segment.length - vehicle_offset_m
        # rescale density
        if rest_segment_length < ending_length:
            n_vehicles_per_mile = n_vehicles * mile / ending_length
        else:
            n_vehicles_per_mile = n_vehicles * mile / rest_segment_length

        los = float("inf")  # in case the vehicles are stuck in traffic jam
        for (low, high), (m, n) in ranges:
            if n_vehicles_per_mile < high:
                d = high - low  # size of range between two densities
                los = m + ((
                                       n_vehicles_per_mile - low) * 0.2 / d)  # -low => shrink to the size of the range
                break

        # reverse the level of service 1.0 means 100% LoS, but the input table defines it in reverse
        return los if los == float("inf") else 1.0 - los

    def level_of_service_in_time_at_segment(self, datetime, segment):
        return self.level_of_service_in_front_of_vehicle(datetime, segment, -1, 0, None)

    def speed_in_time_at_segment(self, datetime, segment):
        speeds = [fcd.speed for fcd in self.fcd_by_segment.get(segment.id, []) if
                  fcd.datetime == datetime]
        if len(speeds) == 0:
            return None
        return sum(speeds) / len(speeds)

    def get_segment_speed(self, node_from: int, node_to: int) -> SpeedKph:
        speeds = {}
        by_segment = list(self.fcd_by_segment[SegmentId((node_from, node_to))])
        by_segment.sort(key=lambda fcd: fcd.datetime)
        for fcd in by_segment:
            speeds[fcd.vehicle_id] = fcd.speed
        speeds = list(speeds.values())
        if len(speeds) == 0:
            return SpeedKph(float('inf'))
        return SpeedKph(sum(speeds) / len(speeds))

    def __getstate__(self):
        return self.fcd_by_segment

    def __setstate__(self, state):
        self.fcd_by_segment = state

    def drop_old(self, dt_threshold):
        for key in tuple(self.fcd_by_segment.keys()):
            self.fcd_by_segment[key] = [fcd for fcd in self.fcd_by_segment[key] if fcd.datetime >= dt_threshold]
