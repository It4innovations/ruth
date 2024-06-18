import logging
from dataclasses import asdict
from datetime import timedelta

import pandas as pd

from ..data.map import BBox
from ..vehicle import Vehicle

logger = logging.getLogger(__name__)


def load_vehicles(input_path: str):
    logger.info("Loading data ... %s", input_path)
    df = pd.read_parquet(input_path, engine="fastparquet")
    vehicles = [Vehicle(
        id=row["id"],
        time_offset=row["time_offset"],
        frequency=row["frequency"],
        start_index=row["start_index"],
        start_distance_offset=row["start_distance_offset"],
        origin_node=row["origin_node"],
        dest_node=row["dest_node"],
        osm_route=row["osm_route"],
        active=row["active"],
        fcd_sampling_period=row["fcd_sampling_period"],
        status=row["status"],

    ) for (_, row) in df.iterrows()]

    bbox_lat_max = df["bbox_lat_max"].iloc[0]
    bbox_lon_min = df["bbox_lon_min"].iloc[0]
    bbox_lat_min = df["bbox_lat_min"].iloc[0]
    bbox_lon_max = df["bbox_lon_max"].iloc[0]
    download_date = df["download_date"].iloc[0]
    bbox = BBox(bbox_lat_max, bbox_lon_min, bbox_lat_min, bbox_lon_max)
    return vehicles, bbox, download_date


def save_vehicles(vehicles, output_path: str):
    logger.info("Saving vehicles ... %s", output_path)

    df = pd.DataFrame([asdict(v) for v in vehicles])
    df.to_pickle(output_path)