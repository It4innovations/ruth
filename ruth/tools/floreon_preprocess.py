import os

import h5py
import numpy as np
from datetime import datetime
import tqdm
from shapely.geometry import LineString
from shapely.wkt import dumps as wkt_dumps

from ruth.data.map import Map, BBox

CHUNK_SIZE = 1_000_000

def get_max_speed(data):
    maxspeed = data.get("maxspeed")
    if isinstance(maxspeed, list):
        maxspeed = maxspeed[0]
    try:
        maxspeed = float(maxspeed)
    except Exception:
        maxspeed = -1.0
    return maxspeed

def get_highway_type(data):
    highway = data.get("highway")
    if highway is None:
        print(f"Warning: edge has no highway tag", flush=True)
        highway = "unknown"

    if isinstance(highway, list):
        highway = highway[0]

    return highway

def get_geometry(data, graph, u, v):
    geom = data.get("geometry")
    if geom is None:
        geom = LineString([
            (graph.nodes[u]['x'], graph.nodes[u]['y']),
            (graph.nodes[v]['x'], graph.nodes[v]['y'])
        ])
    return geom

def add_to_edge_lookup(graph, lookup=None, unique_edges=None):
    if lookup is None:
        lookup = {}

    count_conflicts = 0
    new_edges = 0
    for u, v, data in tqdm.tqdm(graph.edges(data=True), desc="Building edge lookup"):
        if unique_edges and (u, v) not in unique_edges:
            continue

        maxspeed = get_max_speed(data)
        highway = get_highway_type(data)
        geom = get_geometry(data, graph, u, v)

        if (u, v) not in lookup:
            lookup[(u, v)] = (maxspeed, wkt_dumps(geom), highway)
            new_edges += 1
        else:
            conflict = False
            # check for consistency
            if maxspeed != lookup[(u, v)][0]:
                conflict = True
            if highway != lookup[(u,v)][2]:
                conflict = True
                print(f"Warning: conflicting highway type for edge ({u}, {v}): {lookup[(u, v)][2]} vs {highway}", flush=True)
            if wkt_dumps(geom) != lookup[(u, v)][1]:
                print(f"Warning: conflicting geometry for edge ({u}, {v})", flush=True)
                conflict = True

            if conflict:
                count_conflicts += 1

    return lookup, count_conflicts, new_edges


def preprocess_data_for_floreon(
    input_path="fcd_aggregated.h5",
    output_path="fcd_aggregated_enriched.h5",
    frame_offset=0,
):
    with h5py.File(input_path, "r") as f_in, h5py.File(output_path, "w") as f_out:
        src = f_in["grouped_data"]
        interval = int(f_in.attrs['interval_s'].item())
        print("Input interval (s):", interval)

        bbox = f_in.attrs['bbox']
        bbox = BBox(*bbox)
        download_date = f_in.attrs['download_date']
        graph = Map(bbox, download_date=download_date, with_speeds=True, save_hdf=False).network
        edge_lookup, _, _ = add_to_edge_lookup(graph)

        total_rows = src.shape[0]

        # ---- output dataset spec ----
        dt = np.dtype([
            ("timestamp", "i8"),
            ("node_from", "i8"),
            ("node_to", "i8"),
            ("average_speed", "f8"),
            ("volume", "i4"),
            ("max_speed", "f8"),
            ("los", "i8"),
        ])
        dst = f_out.create_dataset(
            "grouped_enriched",
            shape=(total_rows,),
            dtype=dt,
            chunks=(CHUNK_SIZE,)
        )

        # ---- chunked processing ----
        print(f"Total rows to process: {total_rows:,}\n")
        for start in tqdm.tqdm(range(0, total_rows, CHUNK_SIZE), desc="Processing chunks"):
            end = min(start + CHUNK_SIZE, total_rows)
            block = src[start:end]

            out = np.empty(end - start, dtype=dt)

            sum_speed = block["speed_sum_first"] + block["speed_sum_second"]
            sum_speed = sum_speed * 3.6  # convert from m/s to km/h

            count_speed = block["speed_count_first"] + block["speed_count_second"]
            volume = block["count_unique_first"] + block["count_unique_second"]

            avg_speed = np.divide(
                sum_speed, count_speed,
                out=np.zeros_like(sum_speed, dtype=float),
                where=count_speed > 0
            )

            out["timestamp"] = block["rounded_ts"] * interval + frame_offset
            out["node_from"] = block["node_from"]
            out["node_to"] = block["node_to"]
            out["average_speed"] = avg_speed
            out["volume"] = volume

            max_speed = np.empty(end - start, dtype=float)
            # geom = np.empty(end - start, dtype=h5py.string_dtype(encoding="utf-8"))

            for i, (u, v) in enumerate(zip(out["node_from"], out["node_to"])):
                val = edge_lookup.get((u, v))
                if val is None:
                    raise ValueError("Edge not found in graph")
                else:
                    max_speed[i], _, _ = val

            out["max_speed"] = max_speed
            out["los"] = avg_speed / max_speed * 100.0

            dst[start:end] = out

    print("Done")

    # read from the new file the first and last timestamp
    with h5py.File(output_path, "r") as f_out:
        dst = f_out["grouped_enriched"]
        # get min max for each column
        min_timestamp = np.min(dst["timestamp"])
        max_timestamp = np.max(dst["timestamp"])
        print(f"Output data timestamp range: {datetime.utcfromtimestamp(min_timestamp)} - {datetime.utcfromtimestamp(max_timestamp)}")

def export_geometry_to_h5(input_path="fcd_aggregated.h5",
                          output_path="geometry.h5",
                          ):
    with h5py.File(input_path, "r") as f_in, h5py.File(output_path, "w") as f_out:
        src = f_in["grouped_data"]
        interval = int(f_in.attrs['interval_s'].item())
        print("Input interval (s):", interval)

        bbox = f_in.attrs['bbox']
        bbox = BBox(*bbox)
        download_date = f_in.attrs['download_date']
        graph = Map(bbox, download_date=download_date, with_speeds=True, save_hdf=False).network
        edge_lookup, _, _ = add_to_edge_lookup(graph)

        # get set of unique edges in the source data
        unique_edges = set()
        for u, v in zip(src["node_from"], src["node_to"]):
            unique_edges.add((u, v))

        # create a table node_from, node_to, max_speed, geometry
        edge_dt = np.dtype([
            ("node_from", "i8"),
            ("node_to", "i8"),
            ("max_speed", "f8"),
            ("highway", h5py.string_dtype(encoding="utf-8")),
            ("geometry", h5py.string_dtype(encoding="utf-8")),
        ])

        edge_table = np.empty(len(unique_edges), dtype=edge_dt)

        for i, (u, v) in enumerate(unique_edges):
            val = edge_lookup.get((u, v))
            if val is None:
                raise ValueError("Edge not found in graph")
            else:
                max_speed, geom, highway = val
                edge_table[i]["node_from"] = u
                edge_table[i]["node_to"] = v
                edge_table[i]["max_speed"] = max_speed
                edge_table[i]["highway"] = highway
                edge_table[i]["geometry"] = geom

        f_out.create_dataset("edge_data", data=edge_table, chunks=True)

    # open dataset and print first 5entries
    with h5py.File(output_path, "r") as f_out:
        edge_data = f_out["edge_data"]
        print("First 5 entries in edge_data:")
        for row in edge_data[:5]:
            print(row)


def export_geometry_from_folder(input_folder=""):
    hdf5_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(".h5")]

    unique_edges = set()
    for hdf5_path in hdf5_paths:
        with h5py.File(hdf5_path, "r") as f_in:
            src = f_in["grouped_data"]
            total_pairs = len(src)
            for u, v in tqdm.tqdm(zip(src["node_from"], src["node_to"]), total=total_pairs, desc=f"Collecting edges from {hdf5_path}"):
                unique_edges.add((u, v))


    # get all graphml files in the folder
    paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(".graphml")]

    lookup = {}
    total_conflicts = 0
    total_edges = 0
    for path in paths:
        network = Map(graphml_file=path, with_speeds=False).network
        lookup, count_conflicts, new_edges = add_to_edge_lookup(network, lookup, unique_edges)
        print(f"Processed {path}, found {count_conflicts} conflicts and {new_edges} new edges.", flush=True)
        total_conflicts += count_conflicts
        total_edges += new_edges


    print(f"Total conflicts found across all files: {total_conflicts}", flush=True)
    print(f"Total unique edges processed: {total_edges}", flush=True)
    print(f"Size of lookup table: {len(lookup)}", flush=True)

    pass

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Floreon preprocessing")
    parser.add_argument("--input", required=True, help="Path to fcd_aggregated.h5")
    parser.add_argument("--output", default=None, help="Optional output path")
    parser.add_argument("--offset", type=int, default=0, help="Number of seconds to add to timestamp")

    args = parser.parse_args()

    input_path = args.input

    # If output not specified, create name automatically:
    if args.output is None:
        base = os.path.splitext(input_path)[0]
        output_path = base + "_enriched.h5"
    else:
        output_path = args.output

    # If offset not provided, use default logic:
    frame_offset = args.offset

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Offset: {frame_offset}")

    preprocess_data_for_floreon(
        input_path=input_path,
        output_path=output_path,
        frame_offset=frame_offset,
    )
