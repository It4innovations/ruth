import logging
from dataclasses import asdict
from typing import List, Tuple, Optional

import pandas as pd

from ..data.map import BBox
from ..vehicle import Vehicle

logger = logging.getLogger(__name__)


def load_vehicles(input_path: str) -> Tuple[List[Vehicle], Optional[BBox], Optional[pd.Timestamp]]:
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
            frequency=r.get("frequency"),
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


def save_vehicles(vehicles, output_path: str):
    logger.info("Saving vehicles ... %s", output_path)

    df = pd.DataFrame([asdict(v) for v in vehicles])
    df.to_pickle(output_path)