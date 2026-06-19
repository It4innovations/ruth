import logging
import hashlib
import json
from pathlib import Path
from dataclasses import asdict
from datetime import timedelta
from typing import List, Tuple, Optional

import pandas as pd

from ..data.map import BBox
from ..vehicle import Vehicle, VehicleAlternatives, VehicleRouteSelection

logger = logging.getLogger(__name__)


ALTERNATIVES_BY_INDEX = [
    VehicleAlternatives.DEFAULT,
    VehicleAlternatives.DIJKSTRA_FASTEST,
    VehicleAlternatives.DIJKSTRA_SHORTEST,
    VehicleAlternatives.PLATEAU_FASTEST,
]

ROUTE_SELECTION_BY_INDEX = [
    VehicleRouteSelection.NO_ALTERNATIVE,
    VehicleRouteSelection.FIRST,
    VehicleRouteSelection.RANDOM,
    VehicleRouteSelection.PTDR,
]


def choose_by_stable_ratio(ratios: List[float], choices: List, seed: Optional[int], vehicle_id: int, salt: str):
    digest = hashlib.blake2b(
        f"{seed or 0}:{vehicle_id}:{salt}".encode("utf-8"),
        digest_size=8,
    ).digest()
    value = int.from_bytes(digest, byteorder="big") / float(1 << 64)
    cumulative = 0.0
    for ratio, choice in zip(ratios, choices):
        cumulative += ratio
        if value <= cumulative:
            return choice
    return choices[-1]


def set_vehicle_behavior_stable(vehicle: Vehicle,
                                alternatives_ratio: List[float],
                                route_selection_ratio: List[float],
                                seed: Optional[int]):
    if abs(alternatives_ratio[0] - route_selection_ratio[0]) > 1e-8:
        raise ValueError("Default alternatives ratio must equal no-alternative route selection ratio.")

    vehicle.alternatives = choose_by_stable_ratio(alternatives_ratio, ALTERNATIVES_BY_INDEX,
                                                  seed, vehicle.id, "alternatives")

    if vehicle.alternatives == VehicleAlternatives.DEFAULT:
        vehicle.route_selection = VehicleRouteSelection.NO_ALTERNATIVE
        return

    selection_tail = route_selection_ratio[1:]
    selection_total = sum(selection_tail)
    if selection_total <= 0:
        vehicle.route_selection = VehicleRouteSelection.NO_ALTERNATIVE
        return

    selection_ratios = [ratio / selection_total for ratio in selection_tail]
    vehicle.route_selection = choose_by_stable_ratio(selection_ratios, ROUTE_SELECTION_BY_INDEX[1:],
                                                     seed, vehicle.id, "route-selection")

def vehicle_from_record(record, frequency_default=None, fcd_sampling_period_default=None):
    osm_route = record.get("osm_route")
    origin_node = record.get("origin_node")
    dest_node = record.get("dest_node")
    needs_default_route = not osm_route or len(osm_route) <= 2

    if (not osm_route or len(osm_route) < 2) and origin_node is not None and dest_node is not None:
        osm_route = [origin_node, dest_node]

    vehicle = Vehicle(
        id=record.get("id"),
        time_offset=record.get("time_offset"),
        frequency=frequency_default if frequency_default is not None else record.get("frequency"),
        start_index=record.get("start_index", 0),
        start_distance_offset=record.get("start_distance_offset", 0.0),
        origin_node=origin_node,
        dest_node=dest_node,
        osm_route=osm_route,
        active=record.get("active", True),
        fcd_sampling_period=fcd_sampling_period_default if fcd_sampling_period_default is not None else record.get("fcd_sampling_period"),
        status=record.get("status"),
    )
    vehicle.needs_default_route = needs_default_route
    return vehicle


def load_manifest(dataset_path: Path):
    manifest_path = dataset_path / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as f:
            return json.load(f)
    return None


def metadata_from_manifest(manifest):
    shared = manifest.get("shared_columns", {}) if manifest else {}
    bbox_data = shared.get("bbox", {})
    bbox = BBox(
        bbox_data.get("lat_max"),
        bbox_data.get("lon_min"),
        bbox_data.get("lat_min"),
        bbox_data.get("lon_max"),
    )
    return bbox, shared.get("download_date"), shared


def metadata_from_dataframe(df):
    if df.empty:
        return None, None, {}

    row = df.iloc[0]
    shared = {
        "frequency": row.get("frequency"),
        "fcd_sampling_period": row.get("fcd_sampling_period"),
    }
    bbox = BBox(
        row.get("bbox_lat_max"),
        row.get("bbox_lon_min"),
        row.get("bbox_lat_min"),
        row.get("bbox_lon_max"),
    )
    return bbox, row.get("download_date"), shared


def has_complete_map_metadata(bbox, download_date):
    return (
        bbox is not None
        and bbox.north is not None
        and bbox.west is not None
        and bbox.south is not None
        and bbox.east is not None
        and download_date is not None
    )


def bucket_start_from_path(path: Path):
    if "=" not in path.name:
        return None
    key, value = path.name.split("=", 1)
    if key != "start_bucket_s":
        return None
    return int(value)


class VehicleDatasetSource:
    def __init__(self, input_path: str, alternatives_ratio: List[float],
                 route_selection_ratio: List[float], seed: Optional[int]):
        self.input_path = Path(input_path)
        self.manifest = load_manifest(self.input_path)
        self.alternatives_ratio = alternatives_ratio
        self.route_selection_ratio = route_selection_ratio
        self.seed = seed
        self.bucket_paths = self.discover_bucket_paths()
        self.bucket_index = 0

        if not self.bucket_paths:
            raise ValueError(f"No parquet buckets found in {self.input_path}")

        self.bbox, self.download_date, self.shared_defaults = metadata_from_manifest(self.manifest)
        if not has_complete_map_metadata(self.bbox, self.download_date):
            first_bucket = pd.read_parquet(self.bucket_paths[0], engine="fastparquet")
            self.bbox, self.download_date, fallback_defaults = metadata_from_dataframe(first_bucket)
            manifest_defaults = {
                key: value for key, value in self.shared_defaults.items()
                if value is not None and not pd.isna(value)
            }
            self.shared_defaults = {**fallback_defaults, **manifest_defaults}

        if not has_complete_map_metadata(self.bbox, self.download_date):
            raise ValueError(f"Cannot load map metadata from manifest or parquet columns in {self.input_path}")

    def discover_bucket_paths(self):
        bucket_paths = [
            path for path in self.input_path.iterdir()
            if path.is_dir() and bucket_start_from_path(path) is not None
        ]
        return sorted(bucket_paths, key=bucket_start_from_path)

    def has_next_bucket(self):
        return self.bucket_index < len(self.bucket_paths)

    def next_bucket_start_s(self):
        if not self.has_next_bucket():
            return None
        return bucket_start_from_path(self.bucket_paths[self.bucket_index])

    def load_next_bucket(self) -> List[Vehicle]:
        if not self.has_next_bucket():
            return []

        bucket_path = self.bucket_paths[self.bucket_index]
        self.bucket_index += 1
        logger.info("Loading vehicle bucket %s", bucket_path)
        df = pd.read_parquet(bucket_path, engine="fastparquet")
        if df.empty:
            return []

        frequency_default = timedelta(seconds=self.shared_defaults.get("frequency"))
        fcd_sampling_period_default =  timedelta(seconds=self.shared_defaults.get("fcd_sampling_period"))
        vehicles = []
        for record in df.to_dict(orient="records"):
            vehicle = vehicle_from_record(
                record,
                frequency_default,
                fcd_sampling_period_default
            )
            if vehicle.active and vehicle.osm_route and len(vehicle.osm_route) >= 2:
                set_vehicle_behavior_stable(vehicle, self.alternatives_ratio,
                                            self.route_selection_ratio, self.seed)
                vehicles.append(vehicle)
        logger.info("Loaded %d active vehicles from %s", len(vehicles), bucket_path.name)
        return vehicles


def load_vehicles(input_path: str) -> Tuple[List[Vehicle], Optional[BBox], Optional[str]]:
    logger.info("Loading data... %s", input_path)
    df = pd.read_parquet(input_path, engine="fastparquet")
    if df.empty:
        raise ValueError(f"No vehicle data found in {input_path}")

    records = df.to_dict(orient="records")
    vehicles: List[Vehicle] = []
    for r in records:
        vehicles.append(Vehicle(
            id=r.get("id"),
            time_offset=r.get("time_offset"),
            frequency=timedelta(seconds=5),  # TODO: Placeholder, actual frequency may vary
            start_index=r.get("start_index"),
            start_distance_offset=r.get("start_distance_offset"),
            origin_node=r.get("origin_node"),
            dest_node=r.get("dest_node"),
            osm_route=r.get("osm_route"),
            active=r.get("active", True),
            fcd_sampling_period=r.get("fcd_sampling_period"),
            status=r.get("status"),
        ))

    # Filter vehicles with missing or too-short routes or inactive flag
    pre_count = len(vehicles)
    vehicles = [v for v in vehicles if v.active and v.osm_route and len(v.osm_route) >= 2]
    filtered_count = pre_count - len(vehicles)
    if filtered_count > 0:
        logger.info(f"Filtered {filtered_count} vehicles with too short or missing routes.")

    # Helper to read optional columns
    def _get_col(col_name: str):
        return df[col_name].iloc[0] if col_name in df.columns and not df[col_name].empty else None

    bbox_lat_max = _get_col("bbox_lat_max")
    bbox_lon_min = _get_col("bbox_lon_min")
    bbox_lat_min = _get_col("bbox_lat_min")
    bbox_lon_max = _get_col("bbox_lon_max")
    download_date = _get_col("download_date")
    bbox = BBox(bbox_lat_max, bbox_lon_min, bbox_lat_min, bbox_lon_max)
    return vehicles, bbox, download_date


def save_vehicles(vehicles: List[Vehicle], output_path: str) -> None:
    logger.info("Saving vehicles ... %s", output_path)

    df = pd.DataFrame([asdict(v) for v in vehicles])
    df.to_pickle(output_path)
