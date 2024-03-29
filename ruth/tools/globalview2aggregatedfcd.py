import os
import click
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from tqdm import tqdm

from ..simulator.simulation import FCDRecord
from ..utils import round_datetime
from ..fcd_history import FCDHistory
from ..simulator import Simulation


@dataclass
class Record:
    segment_osm_id: str
    fcd_time_calc: datetime
    segment_length: float
    max_speed: float
    current_speed: float

    def __repr__(self):
        return (f"{self.segment_osm_id};{self.fcd_time_calc.strftime('%Y-%m-%d %H:%M')};"
                f"{self.segment_length};{self.max_speed};{self.current_speed}")


def timed_segment_to_record(dt, node_from, node_to, length, max_speed, aggregated_history):
    current_speed = aggregated_history.speed_in_time_at_segment(dt, node_from, node_to)
    if current_speed is None:
        current_speed = max_speed
    seg_id = f"OSM{node_from}T{node_to}"
    return Record(seg_id, dt, round(length, 2), round(max_speed, 2), round(current_speed, 2))


def create_records(sim_path, round_freq_s):
    sim = Simulation.load(sim_path)
    round_freq = timedelta(seconds=round_freq_s)

    m = sim.routing_map
    segment_data = dict()
    for u, v, data in m.original_network.edges(data=True):
        if "length" in data:
            segment_data[(u, v)] = (data["length"], data['speed_kph'])
        else:
            assert False, "Segment without assigned length!"

    rounded_history = []
    for fcd in sim.history.fcd_history:
        attrs = fcd.__dict__
        dt_rounded = round_datetime(attrs['datetime'], round_freq)
        attrs['datetime'] = dt_rounded
        fcd_rounded = FCDRecord(**attrs)
        rounded_history.append(fcd_rounded)
    print("History rounded")

    sim_start = rounded_history[0].datetime
    sim_end = rounded_history[-1].datetime

    aggregated_history = FCDHistory()
    for fcd in rounded_history:
        aggregated_history.add(fcd)

    records = []
    for (node_from, node_to), (seg_length, seg_speed) in tqdm(segment_data.items(), unit=' segment'):
        for time in pd.date_range(start=sim_start, end=sim_end, freq=round_freq):
            record = timed_segment_to_record(time, node_from, node_to, seg_length, seg_speed, aggregated_history)
            records.append(record)

    return records


def aggregate_to_file(sim_path, round_freq_s, out):
    records = create_records(sim_path, round_freq_s)
    with open(out, "w") as csv:
        csv.write("segment_osm_id;fcd_time_calc;segment_length;max_speed;current_speed\n")
        csv.write("\n".join(map(repr, records)))

    print(f"Aggregated FCDs are written within '{out}'.")


aggregate_cmd = click.Group()


@aggregate_cmd.command()
@click.argument("sim_path", type=click.Path(exists=True))
@click.option("--round-freq-s", type=int, default=300, help="How to round date times. [Default: 300 (5 min)]")
@click.option("--out", type=str, default="out.csv")
def aggregate_globalview(sim_path, round_freq_s, out):
    aggregate_to_file(sim_path, round_freq_s, out)


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
        aggregate_to_file(file_path, round_freq_s, out_file_path)


if __name__ == "__main__":
    aggregate_cmd()
