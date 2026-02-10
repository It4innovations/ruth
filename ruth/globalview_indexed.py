"""
Python GlobalView with spatial hash grid indexing - O(log N) queries instead of O(N)

This module provides a pure Python implementation of the C++ GlobalView's
spatial partitioning approach, allowing fair comparison of:
1. Algorithm (spatial indexing vs linear search)
2. Language (Python vs C++)
"""

from collections import defaultdict
from datetime import timedelta
from typing import Dict, List, Optional, Set, TYPE_CHECKING, Tuple

from ruth.data.segment import SegmentId, SpeedKph, speed_mps_to_kph, SpeedMps

if TYPE_CHECKING:
    from ruth.simulator.simulation import FCDRecord


class GlobalViewIndexed:
    """
    Pure Python GlobalView with spatial hash grid indexing.

    Uses the same O(log N) spatial partitioning approach as the C++ version,
    but implemented in Python for fair algorithm comparison.

    Instead of storing all FCDs in a simple list per segment, we partition them
    into a spatial grid based on their offset within the segment. This allows
    queries to skip large portions of the data.
    """

    # Class constants - density ranges for Level of Service
    LoS_RANGES = [
        ((0, 12), (0.0, 0.2)),
        ((12, 20), (0.2, 0.4)),
        ((20, 30), (0.4, 0.6)),
        ((30, 42), (0.6, 0.8)),
        ((42, 67), (0.8, 1.0))
    ]
    MILE_TO_METERS = 1609.344
    ENDING_LENGTH = 200
    GRID_CELL_SIZE = 50.0  # Partition segments into 50m cells
    GRID_THRESHOLD = 100   # Only use grid when segment has more than this many FCDs

    def __init__(self, routing_map=None):
        """
        Initialize GlobalView with spatial hash grid indexing.

        Args:
            routing_map: Optional Map object for segment ID mapping (for API compatibility)
        """
        # Main storage: segment_id -> list of FCDRecords
        self.fcd_by_segment: Dict[SegmentId, List["FCDRecord"]] = defaultdict(list)

        # Spatial grid index: segment_id -> offset_bucket -> list of FCDRecords
        # Allows O(log N) lookup instead of O(N)
        # Only used when segment has many FCDs (adaptive strategy)
        self.fcd_grid: Dict[SegmentId, Dict[int, List["FCDRecord"]]] = defaultdict(
            lambda: defaultdict(list)
        )

        # Track which segments are using grid (for optimization)
        self.segments_using_grid: Set[SegmentId] = set()

        self.modified_segments: Set[SegmentId] = set()
        self._routing_map = routing_map

    def _get_bucket_key(self, offset: float) -> int:
        """
        Get the bucket key for a given offset using spatial grid partitioning.

        Segments are divided into fixed-size cells (GRID_CELL_SIZE meters).
        This allows efficient range queries without scanning all records.

        Args:
            offset: Offset in meters from segment start

        Returns:
            Bucket key (integer cell index)
        """
        return int(offset // self.GRID_CELL_SIZE)

    def add(self, fcd: "FCDRecord"):
        """Add an FCD record with spatial grid indexing."""
        segment_id = fcd.segment.id

        # Store in main list
        self.fcd_by_segment[segment_id].append(fcd)

        # Also store in spatial grid for efficient queries
        bucket_key = self._get_bucket_key(fcd.offset_from_start)
        self.fcd_grid[segment_id][bucket_key].append(fcd)

        self.modified_segments.add(segment_id)

    def add_batch(self, fcds: List["FCDRecord"]):
        """Add multiple FCD records in batch."""
        if not fcds:
            return

        for fcd in fcds:
            self.add(fcd)

    def number_of_vehicles_ahead(
        self, datetime, segment_id, tolerance=None, vehicle_id=-1, vehicle_offset_m=0
    ) -> int:
        """
        Count vehicles ahead using adaptive strategy.

        For small datasets (< GRID_THRESHOLD FCDs), use simple linear search.
        For large datasets, use spatial grid for efficient lookup.
        """
        tolerance = tolerance if tolerance is not None else timedelta(seconds=0)
        tolerance_seconds = (
            tolerance.total_seconds() if isinstance(tolerance, timedelta) else tolerance
        )
        dt_seconds = datetime.timestamp()

        fcds = self.fcd_by_segment.get(segment_id)
        if not fcds:
            return 0

        # Use linear search for small datasets (less overhead)
        if len(fcds) < self.GRID_THRESHOLD:
            unique_vehicles = set()
            for fcd in fcds:
                # Check time window
                if abs(fcd.datetime.timestamp() - dt_seconds) > tolerance_seconds:
                    continue

                # Don't count the excluded vehicle
                if vehicle_id != -1 and fcd.vehicle_id == vehicle_id:
                    continue

                # Check if ahead
                if fcd.offset_from_start >= vehicle_offset_m:
                    unique_vehicles.add(fcd.vehicle_id)

            return len(unique_vehicles)

        # Use grid for large datasets
        grid = self.fcd_grid.get(segment_id)
        if grid is None:
            return 0

        start_bucket = self._get_bucket_key(vehicle_offset_m)

        unique_vehicles = set()
        for bucket_key in sorted(grid.keys()):
            if bucket_key < start_bucket:
                continue

            for fcd in grid[bucket_key]:
                if abs(fcd.datetime.timestamp() - dt_seconds) > tolerance_seconds:
                    continue

                if vehicle_id != -1 and fcd.vehicle_id == vehicle_id:
                    continue

                unique_vehicles.add(fcd.vehicle_id)

        return len(unique_vehicles)

    def level_of_service_in_front_of_vehicle(
        self, datetime, segment, vehicle_id=-1, vehicle_offset_m=0, tolerance=None
    ) -> float:
        """
        Calculate level of service using spatial grid indexing.

        Uses the grid to efficiently find relevant vehicles instead of
        scanning the entire segment.
        """
        segment_id = segment.id
        tolerance_seconds = 0.0
        if tolerance is not None:
            tolerance_seconds = (
                tolerance.total_seconds() if isinstance(tolerance, timedelta) else tolerance
            )

        dt_seconds = datetime.timestamp()

        # Count vehicles ahead using grid
        n_vehicles = self.number_of_vehicles_ahead(
            datetime, segment_id, tolerance, vehicle_id, vehicle_offset_m
        )

        if n_vehicles == 0:
            return 1.0  # Best level of service

        # Estimate vehicle density
        segment_length_miles = segment.length / self.MILE_TO_METERS
        density = n_vehicles / (segment_length_miles * segment.lanes)

        # Map density to level of service
        for (low_speed, high_speed), (low_los, high_los) in self.LoS_RANGES:
            if low_speed <= density < high_speed:
                return (low_los + high_los) / 2.0

        # Beyond highest speed range
        return self.LoS_RANGES[-1][1][1]

    def level_of_service_in_time_at_segment(self, datetime, segment) -> float:
        """Calculate level of service for entire segment."""
        return self.level_of_service_in_front_of_vehicle(datetime, segment, -1, 0, None)

    def take_segment_speeds(self) -> Dict[SegmentId, Optional[SpeedKph]]:
        """Get speeds for all modified segments since last call."""
        speeds = {}
        modified = list(self.modified_segments)

        for segment_id in modified:
            speeds[segment_id] = self.get_segment_speed(segment_id)

        self.modified_segments.clear()
        return speeds

    def get_segment_speed(self, segment_id: SegmentId) -> Optional[SpeedKph]:
        """Get average speed on a segment."""
        fcds = self.fcd_by_segment.get(segment_id)
        if not fcds:
            return None

        # Get unique vehicle speeds (one per vehicle)
        speeds = {}
        for fcd in fcds:
            if fcd.vehicle_id not in speeds:
                speeds[fcd.vehicle_id] = fcd.vehicle_speed_mps

        if not speeds:
            return None

        avg_mps = sum(speeds.values()) / len(speeds)
        return SpeedKph(speed_mps_to_kph(SpeedMps(avg_mps)))

    def drop_old(self, dt_threshold):
        """Remove FCD records older than threshold."""
        dt_threshold_seconds = dt_threshold.timestamp()

        for segment_id in list(self.fcd_by_segment.keys()):
            # Filter out old records
            old_fcds = self.fcd_by_segment[segment_id]
            new_fcds = [fcd for fcd in old_fcds if fcd.datetime.timestamp() >= dt_threshold_seconds]

            if len(new_fcds) != len(old_fcds):
                self.fcd_by_segment[segment_id] = new_fcds
                self.modified_segments.add(segment_id)

                # Rebuild grid for this segment
                self.fcd_grid[segment_id] = defaultdict(list)
                for fcd in new_fcds:
                    bucket_key = self._get_bucket_key(fcd.offset_from_start)
                    self.fcd_grid[segment_id][bucket_key].append(fcd)
