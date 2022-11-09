"""Prepare the input file from Origin/Destination matrix."""
import os.path

import click
import shapely.wkt
import osmnx as ox
import pandas as pd
import numpy as np
from datetime import timedelta
from functools import partial
from multiprocessing import Pool
from shapely.geometry import Point, MultiPoint, Polygon
from tqdm import tqdm

from ..data.border import Border, BorderType, PolygonBorderDef
from ..data.map import Map

graph = None

def gps_to_nodes(od_for_id, poly_wkt):

    global graph
    if graph is None:
        border_kind = "town"
        data_dir = "./data"
        b_def = PolygonBorderDef(poly_wkt)
        b = Border(f"{b_def.md5()}_{border_kind}", b_def, BorderType.parse(border_kind), data_dir, True)
        m = Map(b, data_dir=data_dir)
        graph = m.network

    id, origin_lon, origin_lat, destination_lon, destination_lat, time_offset = od_for_id

    origin_node_id = ox.nearest_nodes(graph, origin_lon, origin_lat)
    dest_node_id = ox.nearest_nodes(graph, destination_lon, destination_lat)

    return id, origin_node_id, dest_node_id, time_offset

def is_active(row):
    return row["origin_node"] != row["dest_node"]

@click.command()
@click.argument("od-matrix-path", type=click.Path(exists=True))
@click.option("--csv-separator", type=str, default=";")
@click.option("--frequency", type=int, default="20",
              help="Default: 20s. A period in which a vehicle asks for rerouting in seconds.")
@click.option("--fcd-sampling-period", type=int, default="5",
              help="Default: 5s. A period in which FCD is stored. It sub samples the frequency.")
@click.option("--border", type=str,
              help="Polygon specifying an area on map. If None it is used convex hull of O/D points")
@click.option("--border-kind", type=str, default="town",
              help="Default 'town'. A kind of border. It can be [country|county|district|town]")
@click.option("--nproc", type=int, default=1, help="Default 1. Number of used processes.")
@click.option("--data-dir", type=click.Path(), default="./data", help="Default './data'.")
@click.option("--out", type=click.Path(), default="out.parquet", help="Default 'out.parquet'.")
def convert(od_matrix_path, csv_separator, frequency, fcd_sampling_period, border, border_kind, nproc, data_dir, out):
    odm_df = pd.read_csv(od_matrix_path, sep=csv_separator)

    orig_points = odm_df[["origin_lon", "origin_lat"]].apply(lambda p: Point(*p), axis=1)
    dest_points = odm_df[["destination_lon", "destination_lat"]].apply(lambda p: Point(*p), axis=1)

    if not os.path.exists(data_dir) or not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    if border is None:
        points = list(orig_points) + list(dest_points)
        mpt = MultiPoint(points)
        border_poly = Polygon(mpt.convex_hull)
    else:
        border_poly = shapely.wkt.loads(border)

    with Pool(processes=nproc) as p:

        od_nodes = []
        with tqdm(total=len(odm_df)) as pbar:
            for odn in p.imap(partial(gps_to_nodes, poly_wkt=border_poly.wkt), odm_df[["id",
                                                                                       "origin_lon",
                                                                                       "origin_lat",
                                                                                       "destination_lon",
                                                                                       "destination_lat",
                                                                                       "time_offset"]].itertuples(index=False,
                                                                                                                  name=None)):
                od_nodes.append(odn)
                pbar.update()

    od_nodes = sorted(od_nodes, key=lambda id_origin_dest: id_origin_dest[0])

    df = pd.DataFrame(od_nodes, columns=["id", "origin_node", "dest_node", "time_offset"])
    df["active"] = df.apply(is_active, axis=1)

    b_def = PolygonBorderDef(border_poly.wkt)

    df[["start_index", "start_distance_offset",
        "frequency", "fcd_sampling_period",
        "border_id", "border_kind", "border"]] = (
        0, 0.0,
        frequency, fcd_sampling_period,
        f"{b_def.md5()}_{border_kind}", border_kind, border_poly.wkt)
    df["osm_route"] = np.empty((len(odm_df), 0)).tolist()
    df["leap_history"] = np.empty((len(odm_df), 0)).tolist()
    df["status"] = "not_started"

    df[["time_offset", "frequency", "fcd_sampling_period"]] = \
        df[["time_offset", "frequency", "fcd_sampling_period"]].applymap(lambda seconds: timedelta(seconds=seconds))

    df.to_parquet(out, engine="fastparquet")


if __name__ == "__main__":
    convert()
