"""Prepare the input file from Origin/Destination matrix."""
import os.path

import click
import shapely.wkt
import osmnx as ox
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from functools import partial
from multiprocessing import Pool
from shapely.geometry import Point, MultiPoint, Polygon
from tqdm import tqdm

from ..data.border import Border, BorderType, PolygonBorderDef
from ..data.map import Map, BBox

routing_map = None


def create_routing_map(poly_wkt, border_kind, bbox, current_date, data_dir):
    b_def = PolygonBorderDef(poly_wkt)
    border = Border(f"{b_def.md5()}_{border_kind}", b_def, BorderType.parse(border_kind), data_dir, True)
    return Map(border, bbox, download_date=current_date, data_dir=data_dir, save_hdf=False)


def gps_to_nodes_with_shortest_path(od_for_id, poly_wkt, border_kind, bbox, current_date, data_dir):
    global routing_map
    if routing_map is None:
        routing_map = create_routing_map(poly_wkt, border_kind, bbox, current_date, data_dir)

    id, origin_lon, origin_lat, destination_lon, destination_lat, time_offset = od_for_id

    origin_node_id = ox.nearest_nodes(routing_map.network, origin_lon, origin_lat)
    dest_node_id = ox.nearest_nodes(routing_map.network, destination_lon, destination_lat)
    osm_route = routing_map.shortest_path(origin_node_id, dest_node_id)

    return id, origin_node_id, dest_node_id, time_offset, osm_route


def get_active_and_state(row):
    if row["origin_node"] == row["dest_node"]:
        return False, "same origin and destination"
    elif row["osm_route"] is None:
        return False, "no route between origin and destination"
    else:
        return True, "not started"


@click.command()
@click.argument("od-matrix-path", type=click.Path(exists=True), metavar='<OD_MATRIX_PATH>')
@click.argument('bounding-coords', nargs=4, type=float, metavar='<BOUNDING_COORDS - NORTH WEST SOUTH EAST>')
@click.option("--csv-separator", type=str, default=";", help="Default: [';'].")
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
def convert(od_matrix_path, bounding_coords, csv_separator, frequency, fcd_sampling_period, border, border_kind, nproc,
            data_dir, out):
    global routing_map
    odm_df = pd.read_csv(od_matrix_path, sep=csv_separator)

    orig_points = odm_df[["lon_from", "lat_from"]].apply(lambda p: Point(*p), axis=1)
    dest_points = odm_df[["lon_to", "lat_to"]].apply(lambda p: Point(*p), axis=1)

    if not os.path.exists(data_dir) or not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    if border is None:
        points = list(orig_points) + list(dest_points)
        mpt = MultiPoint(points)
        border_poly = Polygon(mpt.convex_hull)
    else:
        border_poly = shapely.wkt.loads(border)

    north, west, south, east = bounding_coords
    bbox = BBox(north, west, south, east)
    current_date = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    routing_map = create_routing_map(border_poly.wkt, border_kind, bbox, current_date, data_dir)

    with Pool(processes=nproc) as p:

        od_nodes = []
        with tqdm(total=len(odm_df)) as pbar:
            for odn in p.imap(partial(gps_to_nodes_with_shortest_path,
                                      poly_wkt=border_poly.wkt, border_kind=border_kind, bbox=bbox,
                                      current_date=current_date, data_dir=data_dir),
                              odm_df[["id",
                                      "lon_from", "lat_from",
                                      "lon_to", "lat_to",
                                      "start_offset_s"]].itertuples(index=False, name=None)):
                od_nodes.append(odn)
                pbar.update()

    od_nodes = sorted(od_nodes, key=lambda id_origin_dest: id_origin_dest[0])

    df = pd.DataFrame(od_nodes, columns=["id", "origin_node", "dest_node", "time_offset", "osm_route"])

    b_def = PolygonBorderDef(border_poly.wkt)

    df[["start_index", "start_distance_offset",
        "frequency", "fcd_sampling_period",
        "border_id", "border_kind", "border", "download_date"]] = (
        0, 0.0,
        frequency, fcd_sampling_period,
        f"{b_def.md5()}_{border_kind}", border_kind, border_poly.wkt, current_date)

    df[["time_offset", "frequency", "fcd_sampling_period"]] = \
        df[["time_offset", "frequency", "fcd_sampling_period"]].applymap(lambda seconds: timedelta(seconds=seconds))

    df["active"], df["status"] = zip(*df.apply(get_active_and_state, axis=1))

    df.to_parquet(out, engine="fastparquet")


if __name__ == "__main__":
    convert()
