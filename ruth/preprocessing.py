"""Preprocessing of input data."""

import numpy as np
import pandas as pd
import osmnx as ox
from dataclasses import fields
from datetime import timedelta
from tqdm import tqdm

from .vehicle import Vehicle
from .data.geopoint import GeoPoint
from .data.border import PolygonBorderDef
from .utils import get_map


# TODO: this method should be removed completely
def assign_border(df: pd.DataFrame) -> pd.DataFrame:
    if "border" in df.columns:
        df["border_id"] = df.border.apply(lambda b: f"custom_{hash(PolygonBorderDef(b))}")
    else:
        # default for antarex benchmark
        df["border_id"] = "CZ010" # TODO: get rid of this
        df["border"] = None

    df["border_kind"] = "district"

    return df


def compute_osm_nodes_id(df: pd.DataFrame) -> pd.DataFrame:
    required = ["lat_from", "lon_from", "lat_to", "lon_to", "border_id"]
    assert all(map(lambda item: item in df.columns, required)), \
      f"To compute osm nodes id the dataframe must contain these items: {required}."

    tqdm.pandas(desc="computing OSM nodes")
    df[["origin_node", "dest_node"]] = df.progress_apply(_compute_osm_nodes,
                                                         axis=1,
                                                         result_type="expand")

    df["active"] = df[["origin_node", "dest_node"]].apply(_is_active, axis=1)

    return df


def prepare_vehicle_state(df: pd.DataFrame) -> pd.DataFrame:
    # rename the columns
    df[["time_offset", "frequency", "fcd_sampling_period"]] = \
      df[["start_offset_s", "route_step_s", "log_step_s"]].applymap(lambda seconds: timedelta(seconds=seconds))

    # set segment position
    df[["start_index", "start_distance_offset"]] = (0, 0.0)

    # each car starts with an empty route, only origin and destination is known at that point
    df["osm_route"] = np.empty((len(df), 0)).tolist()  # empty list

    # a history of a one leap, i.e., a series of smaller steps in simulation
    df["leap_history"] = np.empty((len(df), 0)).tolist()  # empty list

    return df[list(map(lambda field: field.name, fields(Vehicle)))]


def _compute_osm_nodes(row):
    lat_from, lon_from = row["lat_from"], row["lon_from"]
    lat_to, lon_to = row["lat_to"], row["lon_to"]

    start_ = GeoPoint(lat_from, lon_from).point()
    end_ = GeoPoint(lat_to, lon_to).point()

    routing_map = get_map(row["border"], row["border_kind"], name=row["border_id"])
    starting_node = ox.distance.nearest_nodes(
        routing_map.network, start_.x, start_.y)
    destination_node = ox.distance.nearest_nodes(
        routing_map.network, end_.x, end_.y)

    return (starting_node, destination_node)

def _is_active(row):
    return row["origin_node"] != row["dest_node"]
