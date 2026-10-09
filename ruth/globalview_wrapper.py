"""
C++ GlobalView wrapper - provides a drop-in replacement for ruth/globalview.py

This module wraps the C++ globalview_cpp module to maintain API compatibility
while using the optimized C++ implementation under the hood.

"""

from datetime import timedelta
from typing import Dict, List, Optional, Set, TYPE_CHECKING
import math

import numpy as np

from .utils import datetime_to_timestamp

try:
    from globalview_cpp import GlobalView as CppGlobalView
    from globalview_cpp import FCDRecord as CppFCDRecord
    from globalview_cpp import fcd_dtype as FCD_DTYPE
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

        # One structured array crosses the boundary in a single copy instead of
        # a Python CppFCDRecord object per record plus a list conversion.
        arr = np.empty(len(fcds), dtype=FCD_DTYPE)
        for i, fcd in enumerate(fcds):
            arr[i] = (
                datetime_to_timestamp(fcd.datetime),
                fcd.vehicle_id,
                self._segment_id_to_int(fcd.segment.id),
                fcd.offset_from_start,
                fcd.vehicle_speed_mps,
            )

        self._cpp_view.add_batch_array(arr)

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

    def add_batch_columns(self, columns):
        """Batch-add FCDs already held as numeric columns (FCDColumns).

        Same effect as add_batch, without building a CppFCDRecord or an
        FCDRecord per sample. Two things add_batch does per record have to be
        preserved, and both reduce to work per DISTINCT segment or vehicle:

          * every segment touched is marked modified; and
          * when a vehicle's segment CHANGES, the segment it left is marked
            modified too -- otherwise its speed never gets recomputed after the
            last car leaves it.

        `columns.segment_id` already holds routing ids, so no conversion is
        needed on the way into C++.
        """
        n = len(columns)
        if n == 0:
            return

        # Field names come from PYBIND11_NUMPY_DTYPE in globalview_bindings.cpp;
        # assigning by name means a rename there fails here rather than silently
        # writing the wrong column.
        arr = np.empty(n, dtype=FCD_DTYPE)
        arr["datetime_seconds"] = columns.datetime_seconds
        arr["vehicle_id"] = columns.vehicle_id
        arr["segment_id"] = columns.segment_id
        arr["offset_from_start"] = columns.offset_from_start
        arr["vehicle_speed_mps"] = columns.vehicle_speed_mps
        self._cpp_view.add_batch_array(arr)

        # Every segment in the batch is modified. One conversion per distinct
        # routing id rather than one per record.
        for routing_id in np.unique(columns.segment_id):
            self._modified_segments.add(self._routing_map.int_to_segment_id(int(routing_id)))

        # A vehicle's last sample in this batch is the segment it ends on; that
        # is what add_batch would have left in _car_to_segment, since it applies
        # the records in order.
        vehicle_id = np.asarray(columns.vehicle_id)
        segment_id = np.asarray(columns.segment_id)
        order = np.lexsort((np.arange(n), vehicle_id))
        sorted_vehicles = vehicle_id[order]
        is_last = np.empty(n, dtype=bool)
        is_last[:-1] = sorted_vehicles[:-1] != sorted_vehicles[1:]
        is_last[-1] = True
        last_index = order[is_last]

        for i in last_index:
            vid = int(vehicle_id[i])
            new_segment = self._routing_map.int_to_segment_id(int(segment_id[i]))
            old_segment = self._car_to_segment.get(vid)
            if old_segment is not None and old_segment != new_segment:
                self._modified_segments.add(old_segment)
            self._car_to_segment[vid] = new_segment

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
        # Structured array: one contiguous buffer in the pickle instead of a
        # dict per record.
        return {
            'fcds': self._cpp_view.export_all_fcds_array(),
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
        fcds = state['fcds']
        if isinstance(fcds, np.ndarray):
            self._cpp_view.import_all_fcds_array(fcds)
        else:
            # Pickles written before the structured-array format: list of dicts
            arr = np.empty(len(fcds), dtype=FCD_DTYPE)
            for i, d in enumerate(fcds):
                arr[i] = (d['datetime_seconds'], d['vehicle_id'], d['segment_id'],
                          d['offset_from_start'], d['vehicle_speed_mps'])
            self._cpp_view.import_all_fcds_array(arr)

    def set_routing_map(self, routing_map):
        self._routing_map = routing_map

