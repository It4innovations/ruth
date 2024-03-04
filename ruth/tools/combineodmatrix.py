import click
import pandas as pd

@click.command()
@click.argument("rules-path", type=click.Path(exists=True))
@click.option("--separator", type=str, default=";")
@click.option("--output", type=click.Path(), default="combined.csv")
def combine_odmatrix(rules_path, separator, output):
    """
    This script enables combining multiple origin destination matrices csv files into one csv file.
    The OD matrices are combined based on the rules. The rules are a csv file with the following format:
    timestamp;swap;path
    7:00:00;0;5K.csv
    12:35:00;1;10K.csv

    The timestamp is the time when the od matrix should be used - offset since midnight.
    The swap is a boolean value which indicates if the origin and destination should be swapped.
    The path is the path to the od matrix.
    """
    df_rules = pd.read_csv(rules_path, sep=separator)
    assert 'timestamp' in df_rules.columns
    assert 'swap' in df_rules.columns
    assert 'path' in df_rules.columns

    # load all od matrices needed
    od_matrices = {}
    for index, row in df_rules.iterrows():
        path = row['path']
        if path not in od_matrices:
            od_matrix_df = pd.read_csv(path, sep=separator)
            assert 'id' in od_matrix_df.columns
            assert 'lat_from' in od_matrix_df.columns
            assert 'lon_from' in od_matrix_df.columns
            assert 'lat_to' in od_matrix_df.columns
            assert 'lon_to' in od_matrix_df.columns
            assert 'start_offset_s' in od_matrix_df.columns

            od_matrix_df['start_offset_s'] = od_matrix_df['start_offset_s'].astype(int)
            od_matrices[path] = od_matrix_df

    df_rules['timestamp'] = pd.to_timedelta(df_rules['timestamp'])
    df_rules['swap'] = df_rules['swap'].astype(bool)

    df_combined = None
    for index, row in df_rules.iterrows():
        timestamp = row['timestamp']
        swap = row['swap']
        path = row['path']

        df_temp = od_matrices[path].copy()

        if swap:
            df_temp['lat_from'], df_temp['lon_from'], df_temp['lat_to'], df_temp['lon_to'] \
                = df_temp['lat_to'], df_temp['lon_to'], df_temp['lat_from'], df_temp['lon_from']

        df_temp['start_offset_s'] = df_temp['start_offset_s'] + timestamp.total_seconds()

        if df_combined is None:
            df_combined = df_temp
        else:
            df_combined = pd.concat([df_combined, df_temp])

    df_combined['id'] = range(1, len(df_combined) + 1)
    df_combined.to_csv(output, sep=separator, index=False)


if __name__ == "__main__":
    combine_odmatrix()
