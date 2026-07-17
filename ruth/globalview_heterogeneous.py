from collections import defaultdict
from datetime import timedelta
from typing import Dict, List, Optional, Set, TYPE_CHECKING, Tuple
import math

from .data.segment import SegmentId, SpeedKph
from .vehicle_types import DEFAULT_VEHICLE_CLASSES

if TYPE_CHECKING:
    from .simulator.simulation import FCDRecord

JAM_DETECTION_MIN_SEGMENT_LENGTH = 10.0
def _as_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float(str(x))


def _norm_vtype(vt) -> str:
    if isinstance(vt, (bytes, bytearray)):
        return vt.decode("utf-8", errors="ignore")
    return str(vt)


class GlobalView:
    def __init__(self, routing_map=None):
        # segment_id -> list[FCDRecord]
        self.fcd_by_segment: Dict[SegmentId, List["FCDRecord"]] = defaultdict(list)

        # vehicle_id -> latest known segment_id
        self.car_to_segment: Dict[int, SegmentId] = {}
        self.vehicle_types: Dict[int, str] = {}

        # segments updated since last take_segment_speeds()
        self.modified_segments: Set[SegmentId] = set()

    def register_vehicles(self, vehicles):
        for vehicle in vehicles:
            self.vehicle_types[vehicle.id] = _norm_vtype(vehicle.vehicle_type)

    def add(self, fcd: "FCDRecord"):
        seg_id = fcd.segment.id
        self.fcd_by_segment[seg_id].append(fcd)
        self.modified_segments.add(seg_id)

        old_segment = self.car_to_segment.get(fcd.vehicle_id)
        if old_segment is not None and old_segment != seg_id:
            self.modified_segments.add(old_segment)

        self.car_to_segment[fcd.vehicle_id] = seg_id

    def load_ahead(
        self,
        dt,
        segment_id,
        tolerance=None,
        vehicle_id: int = -1,
        vehicle_offset_m=0.0,
    ) -> Tuple[float, float, Dict[int, object]]:
        """
        Returns:
          total_pcu: sum PCU of vehicles ahead
          total_occupied_length: sum (length + standstill_gap) ahead (meters)
          vehicles_ahead: dict vid -> VehicleClassParams
        """
        tolerance = tolerance if tolerance is not None else timedelta(seconds=0)
        t0 = dt - tolerance
        t1 = dt + tolerance

        my_off = _as_float(vehicle_offset_m)

        # Keep only the LATEST FCD per vehicle in the time window,
        # and only if vehicle is still currently on this segment.
        latest_by_vehicle: Dict[int, "FCDRecord"] = {}
        for fcd in self.fcd_by_segment.get(segment_id, []):
            if fcd.datetime < t0 or fcd.datetime > t1:
                continue
            if self.car_to_segment.get(fcd.vehicle_id) != segment_id:
                continue
            prev = latest_by_vehicle.get(fcd.vehicle_id)
            if prev is None or fcd.datetime > prev.datetime:
                latest_by_vehicle[fcd.vehicle_id] = fcd

        vehicles_ahead: Dict[int, object] = {}
        for vid, fcd in latest_by_vehicle.items():
            if vid == vehicle_id:
                continue

            off = _as_float(fcd.offset_from_start)
            if off <= my_off:
                continue

            vt = self.vehicle_types.get(vid, "car")
            params = DEFAULT_VEHICLE_CLASSES.get(vt, DEFAULT_VEHICLE_CLASSES["car"])
            vehicles_ahead[vid] = params

        total_pcu = sum(_as_float(p.pcu) for p in vehicles_ahead.values())
        total_occupied = sum(_as_float(p.length_m + p.standstill_gap_m) for p in vehicles_ahead.values())
        return total_pcu, total_occupied, vehicles_ahead

    def level_of_service_in_front_of_vehicle(
        self,
        dt,
        segment,
        vehicle_id: int = -1,
        vehicle_offset_m=0.0,
        tolerance=None,
        limit_vehicle_count: bool = False,
    ) -> float:
        """
        Continuous LoS in (0.05 .. 1.0], where 1.0 = free flow.
        Returns inf when jam is detected.

        - PCU is used for density (crowdedness).
        - physical occupancy is used for jam detection.
        """
        tolerance = tolerance if tolerance is not None else timedelta(seconds=0)

        off_m = _as_float(vehicle_offset_m)
        sum_pcu, sum_occupied, _vehicles_ahead = self.load_ahead(
            dt, segment.id, tolerance, vehicle_id, off_m
        )

        remaining = _as_float(segment.length) - off_m
        if remaining <= 1e-6:
            return 1.0

        lanes = _as_float(getattr(segment, "lanes", 1) or 1)
        if lanes <= 0:
            lanes = 1.0

        # Jam detection (physical space) — only check at segment start (your intended behavior)
        if limit_vehicle_count and off_m <= 1e-9 and _as_float(segment.length) >= 10.0:
            if sum_occupied > remaining * lanes:
                return float("inf")

        # Density PCU/km/lane
        dens = (sum_pcu / max(remaining, 1e-6)) * 1000.0 / lanes

        # Smooth exp curve: higher density -> lower LoS
        los = math.exp(-dens / 25.0)

        if los < 0.05:
            los = 0.05
        if los > 1.0:
            los = 1.0
        return los

    def level_of_service_in_time_at_segment(self, dt, segment):
        return self.level_of_service_in_front_of_vehicle(dt, segment, -1, 0.0, None)

    def take_segment_speeds(self) -> Dict[SegmentId, Optional[SpeedKph]]:
        speeds: Dict[SegmentId, Optional[SpeedKph]] = {}
        for segment_id in self.modified_segments:
            speeds[segment_id] = self.get_segment_speed(segment_id)
        self.modified_segments.clear()
        return speeds

    def get_segment_speed(self, segment_id: SegmentId) -> Optional[SpeedKph]:
        """
        IMPORTANT:
          - fcd.speed is SpeedMps (m/s)
          - Output must be SpeedKph (kph)
        We take the latest speed per vehicle on that segment, average them, and convert to kph.
        """
        latest_speed_by_vehicle: Dict[int, float] = {}

        by_segment = list(self.fcd_by_segment.get(segment_id, []))
        by_segment.sort(key=lambda fcd: fcd.datetime)

        for fcd in by_segment:
            latest_speed_by_vehicle[fcd.vehicle_id] = _as_float(fcd.vehicle_speed_mps)

        if not latest_speed_by_vehicle:
            return None

        avg_mps = sum(latest_speed_by_vehicle.values()) / len(latest_speed_by_vehicle)
        return SpeedKph(avg_mps * 3.6)

    def drop_old(self, dt_threshold):
        for segment_id, old_fcds in list(self.fcd_by_segment.items()):
            new_fcds = [fcd for fcd in old_fcds if fcd.datetime >= dt_threshold]
            if len(new_fcds) != len(old_fcds):
                self.fcd_by_segment[segment_id] = new_fcds
                self.modified_segments.add(segment_id)
