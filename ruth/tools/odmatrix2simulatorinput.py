"""Prepare the input file from Origin/Destination matrix."""
import os.path

import click
import networkx
import osmnx as ox
import pandas as pd
from datetime import datetime
from datetime import timedelta
from functools import partial
from multiprocessing import Pool
from matplotlib import pyplot as plt
from tqdm import tqdm

from ..data.map import Map, BBox

routing_map = None


def create_routing_map(bbox, download_date, data_dir):
    return Map(bbox, download_date=download_date, data_dir=data_dir, save_hdf=False)


def gps_to_nodes_with_shortest_path(od_for_id, bbox, download_date, data_dir):
    global routing_map
    if routing_map is None:
        routing_map = create_routing_map(bbox, download_date, data_dir)

    id, origin_lon, origin_lat, destination_lon, destination_lat, time_offset = od_for_id

    origin_node_id = ox.nearest_nodes(routing_map.network, origin_lon, origin_lat)
    dest_node_id = ox.nearest_nodes(routing_map.network, destination_lon, destination_lat)
    try:
        osm_route = routing_map.shortest_path(origin_node_id, dest_node_id)
    except networkx.NetworkXNoPath:
        osm_route = None

    return id, origin_node_id, dest_node_id, time_offset, osm_route


def get_active_and_state(row):
    if row["origin_node"] == row["dest_node"]:
        return False, "same origin and destination"
    elif row["osm_route"] is None:
        return False, "no route between origin and destination"
    else:
        return True, "not started"


def plot_origin_destination(df, g):
    fig, ax = ox.plot_graph(g, node_size=0, node_color='black', bgcolor="white", edge_color="black",
                            edge_linewidth=0.3, show=False, close=False)

    ax.scatter(df["lon_from"], df["lat_from"], c="red", s=3)
    ax.scatter(df["lon_to"], df["lat_to"], c="blue", s=3)

    plt.show()


@click.command()
@click.argument("od-matrix-path", type=click.Path(exists=True), metavar='<OD_MATRIX_PATH>')
@click.option("--download-date", type=click.DateTime(), default=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
              help="Default: now. The date for which the map is downloaded.")
@click.option("--increase-lat", type=float, default=0.0,
              help="A fraction of the area increase in latitude. (0.1 means 10% increase)")
@click.option("--increase-lon", type=float, default=0.0,
              help="A fraction of the area increase in longitude. (0.1 means 10% increase)")
@click.option("--csv-separator", type=str, default=";", help="Default: [';'].")
@click.option("--frequency", type=int, default="20",
              help="Default: 20s. A period in which a vehicle asks for rerouting in seconds.")
@click.option("--fcd-sampling-period", type=int, default="5",
              help="Default: 5s. A period in which FCD is stored. It sub samples the frequency.")
@click.option("--nproc", type=int, default=1, help="Default 1. Number of used processes.")
@click.option("--data-dir", type=click.Path(), default="./data", help="Default './data'.")
@click.option("--out", type=click.Path(), default="out.parquet", help="Default 'out.parquet'.")
@click.option("--show-only", is_flag=True, help="Show map with cars without computing output.")
def convert(od_matrix_path, download_date, increase_lat, increase_lon, csv_separator, frequency, fcd_sampling_period, nproc,
            data_dir, out, show_only):
    global routing_map

    if not os.path.exists(data_dir) or not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    odm_df = pd.read_csv(od_matrix_path, sep=csv_separator)

    lat_min = min(odm_df["lat_from"].min(), odm_df["lat_to"].min())
    lat_max = max(odm_df["lat_from"].max(), odm_df["lat_to"].max())
    lon_min = min(odm_df["lon_from"].min(), odm_df["lon_to"].min())
    lon_max = max(odm_df["lon_from"].max(), odm_df["lon_to"].max())

    # Increase the area by the given percentage
    lat_min -= (lat_max - lat_min) * increase_lat
    lat_max += (lat_max - lat_min) * increase_lat
    lon_min -= (lon_max - lon_min) * increase_lon
    lon_max += (lon_max - lon_min) * increase_lon

    lat_min = max(lat_min, -90)
    lat_max = min(lat_max, 90)
    lon_min = max(lon_min, -180)
    lon_max = min(lon_max, 180)

    bbox = BBox(lat_max, lon_min, lat_min, lon_max)
    download_date = download_date.replace(hour=0, minute=0, second=0, microsecond=0)
    download_date = download_date.strftime("%Y-%m-%dT%H:%M:%S")
    routing_map = create_routing_map(bbox, download_date, data_dir)

    if show_only:
        plot_origin_destination(odm_df, routing_map.network)
        return

    with Pool(processes=nproc) as p:

        od_nodes = []
        with tqdm(total=len(odm_df)) as pbar:
            for odn in p.imap(partial(gps_to_nodes_with_shortest_path,
                                      bbox=bbox, download_date=download_date,
                                      data_dir=data_dir),
                              odm_df[["id",
                                      "lon_from", "lat_from",
                                      "lon_to", "lat_to",
                                      "start_offset_s"]].itertuples(index=False, name=None)):
                od_nodes.append(odn)
                pbar.update()

    od_nodes = sorted(od_nodes, key=lambda id_origin_dest: id_origin_dest[0])

    df = pd.DataFrame(od_nodes, columns=["id", "origin_node", "dest_node", "time_offset", "osm_route"])

    df[["start_index", "start_distance_offset",
        "frequency", "fcd_sampling_period", "download_date",
        "bbox_lat_max", "bbox_lon_min", "bbox_lat_min", "bbox_lon_max"]] = (
        0, 0.0, frequency, fcd_sampling_period, download_date, lat_max, lon_min, lat_min, lon_max)

    df[["time_offset", "frequency", "fcd_sampling_period"]] = \
        df[["time_offset", "frequency", "fcd_sampling_period"]].applymap(lambda seconds: timedelta(seconds=seconds))

    df["active"], df["status"] = zip(*df.apply(get_active_and_state, axis=1))

    df.to_parquet(out, engine="fastparquet")


if __name__ == "__main__":
    convert()
