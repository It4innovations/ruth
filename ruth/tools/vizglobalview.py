
import re
import click
import pandas as pd
import osmnx as ox
import matplotlib.pyplot as plt
from datetime import timedelta

from ..globalview import GlobalView
from .. import utils


def count_gt_zero(segment_, count):
    return count > 0


p = re.compile("OSM(?P<from_node>\d+)T(?P<to_node>\d+)")


def parse_segment_nodes(segment_id):
    m = p.match(segment_id)

    return int(m.group("from_node")), int(m.group("to_node"))


@click.command()
@click.argument("gv_path", type=click.Path(exists=True))
@click.argument("border_id", type=str)
@click.argument("border_kind", type=str)
@click.argument("start_offset_s", type=int)
@click.option("--tolerance_s", type=int, default=1, help="+- tolerance from the particular time stamp")
@click.option("--bullet_size_factor", type=int, default=1)
@click.option("--out", type=click.Path(), default="out.png")
def viz(gv_path: str,
        border_id: str,
        border_kind: str,
        start_offset_s: int,
        tolerance_s: int,
        bullet_size_factor, out: str):

    df = pd.read_parquet(gv_path, engine="fastparquet")
    gv = GlobalView(data=df)

    df_ni = df.reset_index()
    start = min(df_ni.timestamp)
    td = timedelta(seconds=start_offset_s)
    tolerance = timedelta(seconds=tolerance_s)

    counts = [(segment, gv.number_of_vehicles_in_time_at_segment(start + td, segment, tolerance))
              for segment in df_ni.segment_id]
    counts = filter(lambda seg_count: count_gt_zero(*seg_count), counts)

    unique = list(set(counts))
    print(unique)
    unique_ids = list(map(lambda v: v[0], unique))

    # works only for already downloaded and stored data
    m = utils.get_map(None, border_kind, name=border_id, on_disk=True)

    fig, ax = ox.plot_graph(m.network, show=False)

    centroids = []
    ids = []
    for u, v, data in m.network.edges(data=True):
        osm_id = f"OSM{u}T{v}"
        if osm_id in unique_ids:
            if "geometry" in data:
                c = data["geometry"].centroid
                centroids.append((c.x, c.y))
                ids.append(osm_id)
            else:
                print(f"The segment '{osm_id}' have no geometry defined.")

    x, y = list(zip(*centroids))
    counts = []
    for id in ids:
        idx = unique_ids.index(id)
        _, count = unique[idx]
        counts.append(count)
    print(counts)

    ax.scatter(x, y, sizes=list(map(lambda x: x * bullet_size_factor, counts)), c="red")

    for i, count in enumerate(counts):
        ax.annotate(f"{count}", (x[i], y[i]), c="lemonchiffon")

    plt.show()


if __name__ == "__main__":
    viz()
