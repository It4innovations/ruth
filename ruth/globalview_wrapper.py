"""
C++ GlobalView wrapper - provides a drop-in replacement for ruth/globalview.py

This module wraps the C++ globalview_cpp module to maintain API compatibility
while using the optimized C++ implementation under the hood.

"""

from datetime import timedelta
from typing import Dict, List, Optional, Set, TYPE_CHECKING
import math

from .utils import datetime_to_timestamp

try:
    from globalview_cpp import GlobalView as CppGlobalView
    from globalview_cpp import FCDRecord as CppFCDRecord
except ImportError as e:
    raise ImportError(
        "C++ GlobalView module (globalview_cpp) not found"
    ) from e

from ruth.data.segment import SegmentId, SpeedKph, speed_mps_to_kph

if TYPE_CHECKING:
    from ruth.simulator.simulation import FCDRecord


class GlobalView:
    """
    C++ GlobalView wrapper - REQUIRES C++ implementation (no Python fallback).
    """

    LoS_RANGES = [
        ((0, 12), (0.0, 0.2)),
        ((12, 20), (0.2, 0.4)),
        ((20, 30), (0.4, 0.6)),
        ((30, 42), (0.6, 0.8)),
        ((42, 67), (0.8, 1.0))
    ]
    MILE_TO_METERS = 1609.344
    ENDING_LENGTH = 200

    def __init__(self, routing_map):
        self._cpp_view = CppGlobalView()
        self._car_to_segment: Dict[int, SegmentId] = {}

        # modified segments are tracked in python,
        # because C++ stores segment_ids as ints not SegmentId
        self._modified_segments: Set[SegmentId] = set()
        self._routing_map = routing_map

    def _segment_id_to_int(self, segment_id: SegmentId) -> int:
        return self._routing_map.segment_id_to_int(segment_id)


    def add(self, fcd: "FCDRecord"):
        # Convert segment ID (which may be a tuple) to int for C++ storage
        segment_id_int = self._segment_id_to_int(fcd.segment.id)

        cpp_fcd = CppFCDRecord(
            datetime_to_timestamp(fcd.datetime),
            fcd.vehicle_id,
            segment_id_int,
            fcd.offset_from_start,
            fcd.vehicle_speed_mps,
        )
        self._cpp_view.add(cpp_fcd)

        self._modified_segments.add(fcd.segment.id)
        old_segment = self._car_to_segment.get(fcd.vehicle_id, None)
        if old_segment is not None:
            if old_segment != fcd.segment.id:
                self._modified_segments.add(old_segment)
                self._car_to_segment[fcd.vehicle_id] = fcd.segment.id
        else:
            self._car_to_segment[fcd.vehicle_id] = fcd.segment.id

    def add_batch(self, fcds: List["FCDRecord"]):
        if not fcds:
            return

        cpp_fcds = []
        for fcd in fcds:
            segment_id_int = self._segment_id_to_int(fcd.segment.id)
            cpp_fcd = CppFCDRecord(
                datetime_to_timestamp(fcd.datetime),
                fcd.vehicle_id,
                segment_id_int,
                fcd.offset_from_start,
                fcd.vehicle_speed_mps,
            )
            cpp_fcds.append(cpp_fcd)

        self._cpp_view.add_batch(cpp_fcds)

        # Track modifications
        for fcd in fcds:
            self._modified_segments.add(fcd.segment.id)
            old_segment = self._car_to_segment.get(fcd.vehicle_id, None)
            if old_segment is not None:
                if old_segment != fcd.segment.id:
                    self._modified_segments.add(old_segment)
                    self._car_to_segment[fcd.vehicle_id] = fcd.segment.id
            else:
                self._car_to_segment[fcd.vehicle_id] = fcd.segment.id

    def number_of_vehicles_ahead(self, datetime, segment_id, tolerance=None, vehicle_id=-1, vehicle_offset_m=0) -> int:
        tolerance = tolerance if tolerance is not None else timedelta(seconds=0)
        tolerance_seconds = tolerance.total_seconds() if isinstance(tolerance, timedelta) else tolerance
        dt_seconds = datetime_to_timestamp(datetime)
        segment_id_int = self._segment_id_to_int(segment_id)

        return self._cpp_view.number_of_vehicles_ahead(dt_seconds, segment_id_int, tolerance_seconds, vehicle_id, vehicle_offset_m)

    def level_of_service_in_front_of_vehicle(self, datetime, segment, vehicle_id=-1, vehicle_offset_m=0, tolerance=None) -> float:
        """
        Calculate level of service in front of a vehicle.

        Args:
            datetime: Reference datetime
            segment: Segment object (must have .id, .length, .lanes attributes)
            vehicle_id: Vehicle to exclude (-1 = count all)
            vehicle_offset_m: Vehicle position on segment (meters)
            tolerance: Time tolerance (timedelta)

        Returns:
            Level of service (0.0 = worst, 1.0 = best)
        """
        dt_seconds = datetime_to_timestamp(datetime)

        segment_id = segment.id
        segment_id_int = self._segment_id_to_int(segment_id)
        tolerance_seconds = 0.0
        if tolerance is not None:
            tolerance_seconds = tolerance.total_seconds() if isinstance(tolerance, timedelta) else tolerance

        return self._cpp_view.level_of_service_in_front_of_vehicle(dt_seconds, segment_id_int, float(segment.length),
            int(segment.lanes), tolerance_seconds, vehicle_id, vehicle_offset_m
        )

    def level_of_service_in_time_at_segment(self, datetime, segment) -> float:
        """Calculate level of service for entire segment."""
        return self.level_of_service_in_front_of_vehicle(datetime, segment, -1, 0, None)

    def take_segment_speeds(self) -> Dict[SegmentId, Optional[SpeedKph]]:
        """ Get speeds for all modified segments since last call."""
        speeds = {}
        modified = list(self._modified_segments)

        for segment_id in modified:
            speeds[segment_id] = self.get_segment_speed(segment_id)

        self._modified_segments.clear()
        return speeds

    def get_segment_speed(self, segment_id: SegmentId) -> Optional[SpeedKph]:
        """Get average speed on a segment."""
        segment_id_int = self._segment_id_to_int(segment_id)
        avg_mps = self._cpp_view.get_segment_speed(segment_id_int)
        if math.isnan(avg_mps):
            return None
        return SpeedKph(speed_mps_to_kph(avg_mps))

    def drop_old(self, dt_threshold):
        """Remove FCD records older than threshold."""
        dt_seconds = datetime_to_timestamp(dt_threshold)
        modified_routing_ids = self._cpp_view.drop_old(dt_seconds)

        for routing_id in modified_routing_ids:
            original_segment_id = self._routing_map.int_to_segment_id(routing_id)
            self._modified_segments.add(original_segment_id)

    def __getstate__(self):
        cpp_fcds = self._cpp_view.export_all_fcds()

        serializable_fcds = []
        for fcd in cpp_fcds:
            serializable_fcds.append({
                'datetime_seconds': fcd.datetime_seconds,
                'vehicle_id': fcd.vehicle_id,
                'segment_id': fcd.segment_id,
                'offset_from_start': fcd.offset_from_start,
                'vehicle_speed_mps': fcd.vehicle_speed_mps,
            })

        return {
            'fcds': serializable_fcds,
            'car_to_segment': self._car_to_segment,
            'modified_segments': self._modified_segments,
        }

    def __setstate__(self, state):
        # TODO: add routing map
        self._routing_map = None
        self._cpp_view = CppGlobalView()

        # Restore Python tracking dicts
        self._car_to_segment = state['car_to_segment']
        self._modified_segments = state['modified_segments']

        # Import FCDs back into C++ implementation
        cpp_fcds = []
        for fcd_dict in state['fcds']:
            cpp_fcd = CppFCDRecord(fcd_dict['datetime_seconds'], fcd_dict['vehicle_id'], fcd_dict['segment_id'],
                fcd_dict['offset_from_start'], fcd_dict['vehicle_speed_mps']
            )
            cpp_fcds.append(cpp_fcd)

        self._cpp_view.import_all_fcds(cpp_fcds)

    def set_routing_map(self, routing_map):
        self._routing_map = routing_map

