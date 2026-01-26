from collections import defaultdict
from datetime import timedelta
from typing import Dict, List, Optional, Set, TYPE_CHECKING

from .data.segment import SegmentId, SpeedKph, speed_mps_to_kph, SpeedMps

if TYPE_CHECKING:
    from .simulator.simulation import FCDRecord


class GlobalView:
    # Class constants - density ranges for Level of Service
    # https://transportgeography.org/contents/methods/transport-technical-economic-performance-indicators/levels-of-service-road-transportation/
    LoS_RANGES = [
        ((0, 12), (0.0, 0.2)),
        ((12, 20), (0.2, 0.4)),
        ((20, 30), (0.4, 0.6)),
        ((30, 42), (0.6, 0.8)),
        ((42, 67), (0.8, 1.0))
    ]
    MILE_TO_METERS = 1609.344
    ENDING_LENGTH = 200

    def __init__(self):
        self.fcd_by_segment: Dict[SegmentId, List[FCDRecord]] = defaultdict(list)

        """
        These variables keep track of which segments have been modified since
        the last call to "take_segment_speeds". We use this to only update segments
        that were modified in the map.
        """
        # vehicle ID -> segment ID
        self.car_to_segment: Dict[int, SegmentId] = {}
        self.modified_segments: Set[SegmentId] = set()

    def add(self, fcd: "FCDRecord"):
        self.fcd_by_segment[fcd.segment.id].append(fcd)
        self.modified_segments.add(fcd.segment.id)
        old_segment = self.car_to_segment.get(fcd.vehicle_id, None)
        if old_segment is not None:
            if old_segment != fcd.segment.id:
                self.modified_segments.add(old_segment)
                self.car_to_segment[fcd.vehicle_id] = fcd.segment.id
        else:
            self.car_to_segment[fcd.vehicle_id] = fcd.segment.id

    def number_of_vehicles_ahead(self, datetime, segment_id, tolerance=None, vehicle_id=-1,
                                 vehicle_offset_m=0):
        # counts all vehicles ahead of the vehicle with vehicle_id at the given segment_id at given time range
        # if case vehicle_id not set, then all vehicles are counted

        tolerance = tolerance if tolerance is not None else timedelta(seconds=0)
        dt_min = datetime - tolerance
        dt_max = datetime + tolerance

        fcd_list = self.fcd_by_segment.get(segment_id)
        if not fcd_list:  # Early exit for empty segments
            return 0

        vehicles = set()
        for fcd in fcd_list:
            if fcd.offset_from_start > vehicle_offset_m and fcd.vehicle_id != vehicle_id:
                if dt_min <= fcd.datetime <= dt_max:
                    vehicles.add(fcd.vehicle_id)
        return len(vehicles)

    def level_of_service_in_front_of_vehicle(self, datetime, segment, vehicle_id=-1,
                                             vehicle_offset_m=0, tolerance=None):
        # Cache segment.id locally to avoid repeated property calls
        segment_id = segment.id
        n_vehicles = self.number_of_vehicles_ahead(datetime, segment_id, tolerance,
                                                   vehicle_id, vehicle_offset_m)

        rest_segment_length = segment.length - vehicle_offset_m

        # Rescale density: use ending_length to avoid massive LoS increase at segment end
        denominator = self.ENDING_LENGTH if rest_segment_length < self.ENDING_LENGTH else rest_segment_length
        n_vehicles_per_mile = n_vehicles * self.MILE_TO_METERS / (denominator * segment.lanes)

        # Find LoS by density range
        los = float("inf")
        for (low, high), (m, n) in self.LoS_RANGES:
            if n_vehicles_per_mile < high:
                d = high - low
                los = m + ((n_vehicles_per_mile - low) * 0.2 / d)
                break

        # Reverse the level of service (1.0 = 100% LoS, but ranges are inverted)
        return los if los == float("inf") else 1.0 - los

    def level_of_service_in_time_at_segment(self, datetime, segment):
        return self.level_of_service_in_front_of_vehicle(datetime, segment, -1, 0, None)

    def take_segment_speeds(self) -> Dict[SegmentId, Optional[SpeedKph]]:
        """
        Returns all segments that have been modified since the last call to this
        method, along with their current speeds.
        """
        speeds = {}
        for segment_id in self.modified_segments:
            speeds[segment_id] = self.get_segment_speed(segment_id)
        self.modified_segments.clear()
        return speeds

    def get_segment_speed(self, segment_id: SegmentId) -> Optional[SpeedKph]:
        speeds = {}
        by_segment = list(self.fcd_by_segment[segment_id])
        by_segment.sort(key=lambda fcd: fcd.datetime)
        for fcd in by_segment:
            speeds[fcd.vehicle_id] = fcd.vehicle_speed_mps
        speeds = list(speeds.values())
        if len(speeds) == 0:
            return None
        return SpeedKph(speed_mps_to_kph(SpeedMps(sum(speeds) / len(speeds))))

    def drop_old(self, dt_threshold):
        for segment_id, old_fcds in self.fcd_by_segment.items():
            new_fcds = [fcd for fcd in old_fcds if fcd.datetime >= dt_threshold]
            if len(new_fcds) != len(old_fcds):
                self.fcd_by_segment[segment_id] = new_fcds
                # If the FCDs for the segments have changed, we need to update modified_segments
                self.modified_segments.add(segment_id)
