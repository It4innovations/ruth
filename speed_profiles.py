# code made from https://github.com/It4innovations/ruth/blob/main/ruth/tools/globalview2aggregatedfcd.py
import click
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from tqdm import tqdm

from ruth.utils import round_datetime
from ruth.simulator import Simulation
from ruth.fcd_history import FCDHistory
from ruth.simulator.simulation import FCDRecord


@dataclass
class Record:
    segment_osm_id: str
    fcd_time_calc: datetime
    segment_length: float
    max_speed: float
    current_speed: float
    no_vehicle: bool = False

    def __repr__(self):
        return (f"{self.segment_osm_id};{self.fcd_time_calc.strftime('%Y-%m-%d %H:%M')};"
                f"{self.segment_length};{self.max_speed};{self.current_speed}")


def timed_segment_to_record(dt, node_from, node_to, length, max_speed, aggregated_history):
    current_speed = aggregated_history.speed_in_time_at_segment(dt, node_from, node_to)
    no_vehicle = False
    if current_speed is None:
        current_speed = max_speed
        no_vehicle = True
    seg_id = f"OSM{node_from}T{node_to}"
    return Record(seg_id, dt, round(length, 2), round(max_speed, 2), round(current_speed, 2), no_vehicle)


def create_records(sim_path, round_freq_s, out):
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

    round_freq_min = int(round_freq_s / 60)

    with open(out, "a") as csv:
        for (node_from, node_to), (seg_length, seg_speed) in tqdm(segment_data.items(), unit=' segment'):
            for time in pd.date_range(start=sim_start, end=sim_end, freq=round_freq):
                record = timed_segment_to_record(time, node_from, node_to, seg_length, seg_speed, aggregated_history)

                if not record.no_vehicle:
                    date = record.fcd_time_calc.strftime("%Y-%m-%d")
                    time_in_minutes = record.fcd_time_calc.hour * 60 + record.fcd_time_calc.minute
                    csv.write(f"{date};{record.segment_osm_id};{time_in_minutes};"
                              f"{time_in_minutes + round_freq_min};{int(record.current_speed * 3.6)}\n")


@click.command()
@click.argument("sim_path", type=click.Path(exists=True))
@click.option("--round-freq-s", type=int, default=300, help="How to round date times. [Default: 300 (5 min)]")
@click.option("--out", type=str, default="out.csv")
def aggregate_speed_profiles(sim_path, round_freq_s, out):
    with open(out, "w") as csv:
        csv.write("date;road_id;time_in_minutes_from;time_in_minutes_to;speed_kph\n")

    create_records(sim_path, round_freq_s, out)
    print(f"Aggregated FCDs are written within '{out}'.")


if __name__ == "__main__":
    aggregate_speed_profiles()
