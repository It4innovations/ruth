
import os
import click
import pandas as pd
from ruth.preprocessing import *

@click.command()
@click.argument("in_path", type=click.Path(exists=True))
@click.option("--out", type=str, default="out.pickle")
def main(in_path, out):
    out_path = os.path.abspath(out)
    df = pd.read_csv(in_path, delimiter=';')
    df = assign_border(df)
    df = compute_osm_nodes_id(df)
    df = prepare_vehicle_state(df)
    df.to_pickle(out_path)
    print(f"preprocessed data stored in: {out_path}")

if __name__ == "__main__":
    main()
