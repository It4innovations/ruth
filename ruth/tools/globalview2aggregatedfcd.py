import os
import click
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from tqdm import tqdm

from probduration import Segment

from ..simulator.simulation import FCDRecord
from ..utils import round_datetime
from ..globalview import GlobalView
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


# TODO: update for new global view format
def timed_segment_to_record(dt, node_from, node_to, length, max_speed, aggregated_gv):
    seg_id = f"OSM{node_from}T{node_to}"
    seg = Segment(seg_id, length, max_speed)
    current_speed = aggregated_gv.speed_in_time_at_segment(dt, seg)
    if current_speed is None:
        current_speed = max_speed
    return Record(seg_id, dt, round(length, 2), round(max_speed, 2), round(current_speed, 2))


def aggregate(sim_path, round_freq_s, out=None):
    sim = Simulation.load(sim_path)
    round_freq = timedelta(seconds=round_freq_s)

    m = sim.routing_map
    segment_data = dict()
    for u, v, data in m.original_network.edges(data=True):
        if "length" in data:
            segment_data[(u, v)] = (data["length"], data['speed_kph'])
        else:
            assert False, "Segment without assigned length!"

    aggregated_gv = GlobalView()
    for fcd_data in sim.history.fcd_history:
        dt = round_datetime(fcd_data.datetime, round_freq)
        fcd_rounded = FCDRecord(dt, fcd_data.vehicle_id,
                                fcd_data.segment, fcd_data.start_offset,
                                fcd_data.speed, fcd_data.status, fcd_data.active)
        aggregated_gv.add(fcd_rounded)
    print("History rounded")

    sim_start = sim.history.fcd_history[0].datetime
    sim_start = round_datetime(sim_start, round_freq)
    sim_end = sim.history.fcd_history[-1].datetime
    sim_end = round_datetime(sim_end, round_freq)

    records = []
    for (node_from, node_to), (seg_length, seg_speed) in tqdm(segment_data.items(), unit=' segment'):
        for time in pd.date_range(start=sim_start, end=sim_end, freq=round_freq):
            record = timed_segment_to_record(time, node_from, node_to, seg_length, seg_speed, aggregated_gv)
            records.append(record)

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
