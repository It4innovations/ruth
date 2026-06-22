"""Prepare the input file from Origin/Destination matrix."""
import json
import logging
import os.path
from pathlib import Path
from typing import Dict, Iterable, List

import click
import fastparquet
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

logger = logging.getLogger(__name__)
routing_map = None

INPUT_COLUMNS = ["id", "lon_from", "lat_from", "lon_to", "lat_to", "start_offset_s"]
OUTPUT_SCHEMA_VERSION = 2


def configure_logging():
    if logging.getLogger().handlers:
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s:%(levelname)-4s %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
    )


def create_routing_map(bbox=None, download_date=None, data_dir="./data", map_graphml=None):
    if map_graphml is not None:
        return Map(graphml_file=map_graphml, data_dir=data_dir, save_hdf=False)
    return Map(bbox, download_date=download_date, data_dir=data_dir, save_hdf=False)


def gps_to_nodes_with_shortest_path(od_for_id, bbox, download_date, data_dir,
                                    no_routing=False, map_graphml=None):
    global routing_map
    if routing_map is None:
        routing_map = create_routing_map(bbox=bbox, download_date=download_date,
                                         data_dir=data_dir, map_graphml=map_graphml)

    id, origin_lon, origin_lat, destination_lon, destination_lat, time_offset = od_for_id

    origin_node_id = ox.nearest_nodes(routing_map.network, origin_lon, origin_lat)
    dest_node_id = ox.nearest_nodes(routing_map.network, destination_lon, destination_lat)

    if no_routing:
        osm_route = [origin_node_id, dest_node_id]
    else:
        try:
            osm_route = routing_map.fastest_path(origin_node_id, dest_node_id)
        except networkx.NetworkXNoPath:
            osm_route = None

    return id, origin_node_id, dest_node_id, time_offset, osm_route


def get_active_and_state(row):
    return get_active_and_state_values(row["origin_node"], row["dest_node"], row["osm_route"])


def get_active_and_state_values(origin_node, dest_node, osm_route):
    if origin_node == dest_node:
        return False, "same origin and destination"
    if not osm_route or len(osm_route) < 2:
        return False, "no route between origin and destination"

    return True, "not started"


def plot_origin_destination(df, g):
    fig, ax = ox.plot_graph(g, node_size=0, node_color='black', bgcolor="white", edge_color="black",
                            edge_linewidth=0.3, show=False, close=False)

    ax.scatter(df["lon_from"], df["lat_from"], c="red", s=3)
    ax.scatter(df["lon_to"], df["lat_to"], c="blue", s=3)

    plt.show()


def read_od_chunks(od_matrix_path: str, csv_separator: str, chunk_size: int):
    return pd.read_csv(od_matrix_path, sep=csv_separator, usecols=INPUT_COLUMNS, chunksize=chunk_size)


def compute_input_metadata(od_matrix_path: str, csv_separator: str, chunk_size: int,
                           lat_min_arg: float, lat_max_arg: float,
                           lon_min_arg: float, lon_max_arg: float) -> Dict:
    total_rows = 0
    lat_min = lat_min_arg
    lat_max = lat_max_arg
    lon_min = lon_min_arg
    lon_max = lon_max_arg
    start_offset_min = None
    start_offset_max = None

    for chunk in read_od_chunks(od_matrix_path, csv_separator, chunk_size):
        total_rows += len(chunk)
        lat_min = min(chunk["lat_from"].min(), chunk["lat_to"].min(), lat_min)
        lat_max = max(chunk["lat_from"].max(), chunk["lat_to"].max(), lat_max)
        lon_min = min(chunk["lon_from"].min(), chunk["lon_to"].min(), lon_min)
        lon_max = max(chunk["lon_from"].max(), chunk["lon_to"].max(), lon_max)

        chunk_start_min = chunk["start_offset_s"].min()
        chunk_start_max = chunk["start_offset_s"].max()
        start_offset_min = chunk_start_min if start_offset_min is None else min(start_offset_min, chunk_start_min)
        start_offset_max = chunk_start_max if start_offset_max is None else max(start_offset_max, chunk_start_max)

    if total_rows == 0:
        raise ValueError(f"No OD pairs found in {od_matrix_path}")

    return {
        "total_rows": total_rows,
        "lat_min": lat_min,
        "lat_max": lat_max,
        "lon_min": lon_min,
        "lon_max": lon_max,
        "start_offset_min": int(start_offset_min),
        "start_offset_max": int(start_offset_max),
    }


def expand_bbox(metadata: Dict, increase_lat: float, increase_lon: float) -> Dict:
    lat_min = metadata["lat_min"]
    lat_max = metadata["lat_max"]
    lon_min = metadata["lon_min"]
    lon_max = metadata["lon_max"]

    lat_diff = lat_max - lat_min
    lon_diff = lon_max - lon_min
    lat_min -= lat_diff * increase_lat
    lat_max += lat_diff * increase_lat
    lon_min -= lon_diff * increase_lon
    lon_max += lon_diff * increase_lon

    return {
        "lat_min": max(lat_min, -90),
        "lat_max": min(lat_max, 90),
        "lon_min": max(lon_min, -180),
        "lon_max": min(lon_max, 180),
    }


def bbox_values_from_map(routing_map: Map) -> Dict:
    xs = []
    ys = []
    for _, data in routing_map.network.nodes(data=True):
        if "x" in data and "y" in data:
            xs.append(float(data["x"]))
            ys.append(float(data["y"]))

    if not xs or not ys:
        raise ValueError("Cannot infer bbox from graphml: graph nodes do not contain x/y coordinates.")

    return {
        "lat_min": min(ys),
        "lat_max": max(ys),
        "lon_min": min(xs),
        "lon_max": max(xs),
    }


def get_graph_download_date(routing_map: Map):
    for key in ("download_date", "created_date", "created_at", "osm_date"):
        value = routing_map.network.graph.get(key)
        if value:
            return str(value)
    return None


def normalize_to_list(values):
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, list):
        return values
    if isinstance(values, tuple):
        return list(values)
    return [values]


def gps_to_nodes_no_routing(chunk: pd.DataFrame) -> List:
    origin_node_ids = normalize_to_list(
        ox.nearest_nodes(routing_map.network, chunk["lon_from"].to_numpy(), chunk["lat_from"].to_numpy())
    )
    dest_node_ids = normalize_to_list(
        ox.nearest_nodes(routing_map.network, chunk["lon_to"].to_numpy(), chunk["lat_to"].to_numpy())
    )

    rows = []
    for id_, origin_node_id, dest_node_id, time_offset in zip(
            chunk["id"],
            origin_node_ids,
            dest_node_ids,
            chunk["start_offset_s"]):
        rows.append((id_, origin_node_id, dest_node_id, time_offset,
                     [origin_node_id, dest_node_id]))

    return rows


def gps_to_nodes_routed(chunk: pd.DataFrame, bbox: BBox, download_date: str, data_dir: str,
                        no_routing: bool, pool, map_graphml: str = None) -> List:
    worker_fn = partial(gps_to_nodes_with_shortest_path,
                        bbox=bbox, download_date=download_date,
                        data_dir=data_dir, no_routing=no_routing,
                        map_graphml=map_graphml)
    rows = chunk[["id", "lon_from", "lat_from", "lon_to", "lat_to",
                  "start_offset_s"]].itertuples(index=False, name=None)

    if pool is None:
        return [worker_fn(row) for row in rows]

    return list(pool.imap(worker_fn, rows))


def build_output_dataframe(od_nodes: Iterable, frequency: int, fcd_sampling_period: int,
                           download_date: str, bbox_values: Dict, partition_seconds: int) -> pd.DataFrame:
    df = pd.DataFrame(od_nodes, columns=["id", "origin_node", "dest_node", "time_offset", "osm_route"])

    if df.empty:
        return df

    df = df.sort_values("id").reset_index(drop=True)
    df["time_offset_s"] = df["time_offset"].astype("int64")
    df["start_bucket_s"] = (df["time_offset_s"] // int(partition_seconds)) * int(partition_seconds)

    df[["start_index", "start_distance_offset",
        "frequency", "fcd_sampling_period", "download_date",
        "bbox_lat_max", "bbox_lon_min", "bbox_lat_min", "bbox_lon_max"]] = (
        0, 0.0, frequency, fcd_sampling_period, download_date,
        bbox_values["lat_max"], bbox_values["lon_min"], bbox_values["lat_min"], bbox_values["lon_max"])

    df[["time_offset", "frequency", "fcd_sampling_period"]] = \
        df[["time_offset", "frequency", "fcd_sampling_period"]].map(lambda seconds: timedelta(seconds=seconds))

    states = [
        get_active_and_state_values(origin_node, dest_node, osm_route)
        for origin_node, dest_node, osm_route in zip(df["origin_node"], df["dest_node"], df["osm_route"])
    ]
    df["active"], df["status"] = zip(*states)

    return df[[
        "id", "origin_node", "dest_node", "time_offset", "osm_route",
        "start_index", "start_distance_offset", "frequency", "fcd_sampling_period",
        "download_date", "bbox_lat_max", "bbox_lon_min", "bbox_lat_min", "bbox_lon_max",
        "active", "status", "time_offset_s", "start_bucket_s"
    ]]


def resolve_output_format(out: str, output_format: str) -> str:
    if output_format != "auto":
        return output_format

    out_path = Path(out)
    if out_path.exists() and out_path.is_dir():
        return "dataset"
    if out_path.suffix == ".parquet":
        return "single"
    return "dataset"


def output_specs(out: str, output_format: str):
    if output_format != "both":
        return [(out, output_format)]

    out_path = Path(out)
    if out_path.suffix == ".parquet":
        dataset_path = out_path.with_suffix("")
    else:
        dataset_path = out_path
    return [
        (str(out_path), "single"),
        (str(dataset_path), "dataset"),
    ]


def final_single_output_path(out: str, active_vehicles: int) -> Path:
    out_path = Path(out)
    if out_path.suffix == ".parquet":
        return out_path.with_name(f"{out_path.stem}-{active_vehicles}{out_path.suffix}")
    return out_path.with_name(f"{out_path.name}-{active_vehicles}.parquet")


def temp_single_output_path(out: str) -> Path:
    out_path = Path(out)
    suffix = out_path.suffix or ".parquet"
    return out_path.with_name(f".{out_path.stem or out_path.name}.tmp-{os.getpid()}{suffix}")


def manifest_path_for(output_path: Path, output_format: str) -> Path:
    if output_format == "dataset":
        return output_path / "manifest.json"
    return output_path.with_suffix(".manifest.json")


class ParquetOutputWriter:
    DATASET_MANIFEST_COLUMNS = [
        "frequency",
        "fcd_sampling_period",
    ]

    def __init__(self, out: str, output_format: str):
        self.out = Path(out)
        self.output_format = output_format
        self.rows_written = 0
        self.temp_path = None
        self.final_path = None

        if self.output_format == "single":
            self.temp_path = temp_single_output_path(out)
            self.temp_path.parent.mkdir(parents=True, exist_ok=True)
            if self.temp_path.exists():
                self.temp_path.unlink()
        else:
            self.out.mkdir(parents=True, exist_ok=True)
            existing_entries = list(self.out.iterdir())
            if existing_entries:
                raise FileExistsError(f"Dataset output directory must be empty: {self.out}")

    def write(self, df: pd.DataFrame) -> None:
        if df.empty:
            return

        if self.output_format == "single":
            fastparquet.write(
                str(self.temp_path),
                df,
                append=self.rows_written > 0,
                write_index=False,
            )
        else:
            df = df.drop(columns=self.DATASET_MANIFEST_COLUMNS, errors="ignore")
            fastparquet.write(
                str(self.out),
                df,
                append=self.rows_written > 0,
                file_scheme="hive",
                partition_on=["start_bucket_s"],
                write_index=False,
            )

        self.rows_written += len(df)

    def close(self, active_vehicles: int) -> Path:
        if self.output_format == "single":
            self.final_path = final_single_output_path(str(self.out), active_vehicles)
            self.final_path.parent.mkdir(parents=True, exist_ok=True)
            self.temp_path.replace(self.final_path)
        else:
            self.final_path = self.out
        return self.final_path


def build_manifest(od_matrix_path: str, output_path: Path, output_format: str, metadata: Dict,
                   bbox_values: Dict, download_date: str, frequency: int, fcd_sampling_period: int,
                   active_vehicles: int, inactive_vehicles: int, no_routing: bool,
                   chunk_size: int, partition_seconds: int, map_graphml: str = None,
                   download_date_source: str = "option") -> Dict:
    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "generator": "ruth.tools.odmatrix2simulatorinput",
        "input": {
            "path": str(od_matrix_path),
            "rows": int(metadata["total_rows"]),
            "start_offset_s_min": int(metadata["start_offset_min"]),
            "start_offset_s_max": int(metadata["start_offset_max"]),
        },
        "output": {
            "path": str(output_path),
            "format": output_format,
            "rows": int(active_vehicles + inactive_vehicles),
            "active_vehicles": int(active_vehicles),
            "inactive_vehicles": int(inactive_vehicles),
            "chunk_size": int(chunk_size),
            "partition_seconds": int(partition_seconds),
        },
        "routing": {
            "mode": "no-routing" if no_routing else "shortest-path",
            "osm_route": "[origin_node, dest_node]" if no_routing else "fastest_path",
        },
        "map": {
            "source": "graphml" if map_graphml else "bbox-download",
            "graphml_file": str(map_graphml) if map_graphml else None,
            "download_date_source": download_date_source,
        },
        "shared_columns": {
            "frequency": int(frequency),
            "fcd_sampling_period": int(fcd_sampling_period),
            "download_date": download_date,
            "bbox": {
                "lat_max": float(bbox_values["lat_max"]),
                "lon_min": float(bbox_values["lon_min"]),
                "lat_min": float(bbox_values["lat_min"]),
                "lon_max": float(bbox_values["lon_max"]),
            },
        },
    }


def write_manifest(manifest: Dict, output_path: Path, output_format: str) -> Path:
    manifest_path = manifest_path_for(output_path, output_format)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    return manifest_path


@click.command()
@click.argument("od-matrix-path", type=click.Path(exists=True), metavar='<OD_MATRIX_PATH>')
@click.option("--download-date", type=click.DateTime(), default=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
              help="Default: now. The date for which the map is downloaded.")
@click.option("--increase-lat", type=float, default=0.0,
              help="A fraction of the area increase in latitude. (0.1 means 10% increase)")
@click.option("--increase-lon", type=float, default=0.0,
              help="A fraction of the area increase in longitude. (0.1 means 10% increase)")
@click.option("--lat-min", type=float, default=90, help="Set to increase map size.")
@click.option("--lat-max", type=float, default=-90, help="Set to increase map size.")
@click.option("--lon-min", type=float, default=180, help="Set to increase map size.")
@click.option("--lon-max", type=float, default=-180, help="Set to increase map size.")
@click.option("--csv-separator", type=str, default=";", help="Default: [';'].")
@click.option("--frequency", type=int, default=20,
              help="Default: 20s. A period in which a vehicle asks for rerouting in seconds.")
@click.option("--fcd-sampling-period", type=int, default=5,
              help="Default: 5s. A period in which FCD is stored. It sub samples the frequency.")
@click.option("--nproc", type=int, default=1, help="Default 1. Number of used processes.")
@click.option("--data-dir", type=click.Path(), default="./data", help="Default './data'.")
@click.option("--map-graphml", type=click.Path(exists=True),
              help="Load routing map from this GraphML file instead of downloading/loading by bbox and date.")
@click.option("--out", type=click.Path(), default="out.parquet", help="Default 'out.parquet'.")
@click.option("--show-only", is_flag=True, help="Show map with cars without computing output.")
@click.option("--no-routing", is_flag=True, help="Skip route calculation; osm_route will be [origin_node, dest_node].")
@click.option("--chunk-size", type=int, default=200_000,
              help="Default: 200000. Number of OD rows processed per chunk.")
@click.option("--partition-seconds", type=int, default=900,
              help="Default: 900. Start-time bucket size for dataset output.")
@click.option("--output-format", type=click.Choice(["auto", "single", "dataset", "both"]), default="auto",
              help="Default: auto. 'single' writes one parquet file; 'dataset' writes a parquet directory; "
                   "'both' writes both for comparison.")
def convert(od_matrix_path, download_date, increase_lat, increase_lon,
            lat_min, lat_max, lon_min, lon_max,
            csv_separator, frequency, fcd_sampling_period, nproc,
            data_dir, map_graphml, out, show_only, no_routing, chunk_size,
            partition_seconds, output_format):
    global routing_map
    configure_logging()

    if not os.path.exists(data_dir) or not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    if chunk_size <= 0:
        raise ValueError("--chunk-size must be greater than 0")
    if partition_seconds <= 0:
        raise ValueError("--partition-seconds must be greater than 0")

    logger.info("Scanning OD matrix from '%s' in chunks of %d ...", od_matrix_path, chunk_size)
    metadata = compute_input_metadata(od_matrix_path, csv_separator, chunk_size,
                                      lat_min, lat_max, lon_min, lon_max)
    logger.info("Found %d OD pairs.", metadata["total_rows"])

    download_date = download_date.replace(hour=0, minute=0, second=0, microsecond=0)
    download_date = download_date.strftime("%Y-%m-%dT%H:%M:%S")
    download_date_source = "option"

    if map_graphml:
        logger.info("Loading routing map from GraphML '%s' ...", map_graphml)
        routing_map = create_routing_map(data_dir=data_dir, map_graphml=map_graphml)
        bbox_values = bbox_values_from_map(routing_map)
        graph_download_date = get_graph_download_date(routing_map)
        if graph_download_date:
            download_date = graph_download_date
            download_date_source = "graphml"
        bbox = BBox(bbox_values["lat_max"], bbox_values["lon_min"],
                    bbox_values["lat_min"], bbox_values["lon_max"])
        logger.info("Using bbox inferred from GraphML nodes.")
    else:
        logger.info("Computing bounding box ...")
        bbox_values = expand_bbox(metadata, increase_lat, increase_lon)
        bbox = BBox(bbox_values["lat_max"], bbox_values["lon_min"],
                    bbox_values["lat_min"], bbox_values["lon_max"])
        logger.info("Loading routing map (date: %s) ...", download_date)
        routing_map = create_routing_map(bbox=bbox, download_date=download_date, data_dir=data_dir)

    lat_min = bbox_values["lat_min"]
    lat_max = bbox_values["lat_max"]
    lon_min = bbox_values["lon_min"]
    lon_max = bbox_values["lon_max"]
    logger.info("BBox: lat=[%.4f, %.4f], lon=[%.4f, %.4f]", lat_min, lat_max, lon_min, lon_max)
    logger.info("Map loaded: %d nodes, %d edges.",
                len(routing_map.network.nodes), len(routing_map.network.edges))

    if show_only:
        odm_df = pd.concat(read_od_chunks(od_matrix_path, csv_separator, chunk_size), ignore_index=True)
        plot_origin_destination(odm_df, routing_map.network)
        return

    routing_label = "nearest nodes only (no routing)" if no_routing else "dijkstra fastest paths"
    output_format = resolve_output_format(out, output_format)
    writers = [
        ParquetOutputWriter(output_path, writer_format)
        for output_path, writer_format in output_specs(out, output_format)
    ]
    pool = None
    active_vehicles = 0
    inactive_vehicles = 0

    logger.info("Computing %s for %d OD pairs using %d processes ...",
                routing_label, metadata["total_rows"], nproc)
    logger.info("Writing %s parquet output incrementally ...", output_format)

    try:
        if not no_routing and nproc > 1:
            pool = Pool(processes=nproc)

        with tqdm(total=metadata["total_rows"], desc="Routing", unit="pair") as pbar:
            for chunk in read_od_chunks(od_matrix_path, csv_separator, chunk_size):
                if no_routing:
                    od_nodes = gps_to_nodes_no_routing(chunk)
                else:
                    od_nodes = gps_to_nodes_routed(chunk, bbox, download_date, data_dir,
                                                   no_routing, pool, map_graphml=map_graphml)

                df = build_output_dataframe(od_nodes, frequency, fcd_sampling_period,
                                            download_date, bbox_values, partition_seconds)
                chunk_active = int(df["active"].sum())
                active_vehicles += chunk_active
                inactive_vehicles += len(df) - chunk_active
                for writer in writers:
                    writer.write(df)
                pbar.update(len(chunk))
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    logger.info("Active vehicles:   %d", active_vehicles)
    logger.info("Inactive vehicles: %d (same origin/destination or no route found)", inactive_vehicles)

    for writer in writers:
        final_output = writer.close(active_vehicles)
        manifest = build_manifest(od_matrix_path, final_output, writer.output_format, metadata,
                                  bbox_values, download_date, frequency, fcd_sampling_period,
                                  active_vehicles, inactive_vehicles, no_routing,
                                  chunk_size, partition_seconds,
                                  map_graphml=map_graphml,
                                  download_date_source=download_date_source)
        manifest_path = write_manifest(manifest, final_output, writer.output_format)
        logger.info("Output saved to '%s'.", final_output)
        logger.info("Manifest saved to '%s'.", manifest_path)


if __name__ == "__main__":
    convert()
