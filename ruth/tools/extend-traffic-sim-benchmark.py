
import os
import click
import pandas as pd


@click.command()
@click.argument("in-path", type=click.Path(exists=True))
@click.argument("n-times", type=int)
@click.option("--out", type=str, default="out-extend.csv")
def extend(in_path, n_times, out):
    out_path = os.path.abspath(out)
    df = pd.read_csv(in_path, delimiter=";")

    nrows = len(df)
    for _, row in df.iterrows():
        for i in range(1, n_times):
            s = row.copy()
            s["id"] = i * nrows + s["id"]
            df = df.append(s, ignore_index=True)

    df = df.sort_values(by=["id"])
    df.to_csv(out_path, sep=";")


if __name__ == "__main__":
    extend()
