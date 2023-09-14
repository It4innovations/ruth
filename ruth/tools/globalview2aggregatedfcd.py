
import os
import click
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from tqdm import tqdm

from probduration import Segment

from ..utils import round_datetime
from ..globalview import GlobalView
from ..simulator import Simulation


@dataclass
class Record:
    segment_osm_id: str
    fcd_time_calc: datetime
    los: float
    segment_length: float

    def __repr__(self):
        return f"{self.segment_osm_id};{self.fcd_time_calc.strftime('%Y-%m-%d %H:%M')};{self.los};{self.segment_length}"


def timed_segment_to_record(dt, seg_id, length, aggregated_gv):
    seg = Segment(seg_id, length, 0)  # NOTE: 0 is the max. allowed speed, which is not used in this context.
    los = aggregated_gv.level_of_service_in_time_at_segment(dt, seg)
    if float('inf') == los:
        los = 0.0
    return Record(seg_id, dt, los, length)


def aggregate(sim_path, round_freq_s, out=None):
    sim = Simulation.load(sim_path)
    round_freq = timedelta(seconds=round_freq_s)

    # collect lengths of the segments
    vehicle_representative = sim.vehicles[0]
    df = pd.DataFrame(sim.history.data,
                      columns=["timestamp", "segment_id", "vehicle_id",
                               "start_offset", "speed", "segment_length",
                               "status"])

    df_ni = df.reset_index()
    segment_ids = df_ni["segment_id"].unique()
    m = sim.routing_map
    segment_lengths = dict()
    for u, v, data in m.network.edges(data=True):
        osm_id = f"OSM{u}T{v}"
        if osm_id in segment_ids:
            if "length" in data:
                segment_lengths[osm_id] = data["length"]
            else:
                assert False, "Segment without assigned length!"

    rounded_history = []
    for dt, *vals in sim.history.data:
        dt_rounded = round_datetime(dt, round_freq)
        rounded_history.append((dt_rounded, *vals))

    print("History rounded")
    aggregated_gv = GlobalView(data=rounded_history)
    unique_segments_in_time = [(row[0], row[1], segment_lengths[row[1]]) for row in aggregated_gv.data]
    unique_segments_in_time = list(dict.fromkeys(unique_segments_in_time))

    print(f"Computing the aggregation for {len(unique_segments_in_time)} items...")
    # records = map(partial(timed_segment_to_record, aggregated_gv=aggregated_gv), unique_segments_in_time)
    records = []
    for dt, seg_id, length in tqdm(unique_segments_in_time):
        records.append(timed_segment_to_record(dt, seg_id, length, aggregated_gv))
    print("Data aggregated.")

    with open(out, "w") as csv:
        csv.write("segment_osm_id;fcd_time_calc;los;segment_length\n")
        csv.write("\n".join(map(repr, records)))

    print(f"Aggregated FCDs are written within '{out}'.")


aggregate_cmd = click.Group()


@aggregate_cmd.command()
@click.argument("sim_path", type=click.Path(exists=True))
@click.option("--round-freq-s", type=int, default=300, help="How to round date times. [Default: 300 (5 min)]")
@click.option("--out", type=str, default="out.csv")
def aggregate_globalview(sim_path, round_freq_s, out):
    return aggregate(sim_path, round_freq_s, out)


@aggregate_cmd.command()
@click.argument("dir_path", type=click.Path(exists=True))
@click.option("--round-freq-s", type=int, default=300, help="How to round date times [Default: 300 (5 min)].")
@click.option("--out-dir", type=str, default="out")
def aggregate_globalview_set(dir_path, round_freq_s, out_dir):
    dir_path_ = os.path.abspath(dir_path)
    out_dir_ = os.path.abspath(out_dir)

    if not os.path.exists(out_dir_):
        os.makedirs(out_dir_)

    for filename in os.listdir(dir_path_):
        file_path = os.path.join(dir_path_, filename)
        basename = os.path.basename(filename)
        parts = basename.split('.')
        name = parts[0]

        out_file_path = os.path.join(out_dir_, f"{name}-aggregated-fcd.csv")
        aggregate(file_path, round_freq_s, out_file_path)


if __name__ == "__main__":
    aggregate_cmd()








