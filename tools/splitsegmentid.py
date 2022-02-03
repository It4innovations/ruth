
import os
import re
import click
import pandas as pd

@click.command()
@click.argument("global_view", type=click.Path(exists=True))
@click.option("--out", type=str, default="out.pickle")
def main(global_view, out):
    out_path = os.path.abspath(out)
    df = pd.read_pickle(global_view)
    index_columns = df.index.names

    df.reset_index(inplace=True)
    osm_id_regex = re.compile("OSM(?P<node_from>\d+)T(?P<node_to>\d+)")
    df[["node_from", "node_to"]] = df["segment_id"].apply(
        lambda seg_id: osm_id_regex.match(seg_id).groups()).tolist()

    df = df.loc[:, df.columns != "segment_id"] # remove column with segment_id

    if "segment_id" in index_columns:
        index_columns = list(filter(lambda v: v != "segment_id", index_columns))
        index_columns += ["node_from", "node_to"]
    df.set_index(index_columns, inplace=True)
    df.sort_index(inplace=True)

    df.to_pickle(out_path)
