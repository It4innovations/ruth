
import click
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

from probduration import Segment

from ..utils import get_map, round_datetime
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


def aggregate(sim_path, round_freq_s, border_id=None, border_kind=None, out=None):
    sim = Simulation.load(sim_path)
    round_freq = timedelta(seconds=round_freq_s)

    # collect lengths of the segments
    vehicle_representative = sim.vehicles[0]
    border_id_ = vehicle_representative.border_id if border_id is None else border_id
    border_kind_ = vehicle_representative.border_kind if border_kind is None else border_kind
    df = pd.DataFrame(sim.history.data,
                      columns=["timestamp", "segment_id", "vehicle_id", "start_offset", "speed", "status"])

    df_ni = df.reset_index()
    segment_ids = df_ni["segment_id"].unique()
    m = get_map(None, border_kind_, name=border_id_, on_disk=True)
    segment_lengths = dict()
    for u, v, data in m.network.edges(data=True):
        osm_id = f"OSM{u}T{v}"
        if osm_id in segment_ids:
            if "length" in data:
                segment_lengths[osm_id] = data["length"]
            else:
                assert False, "Segment without assigned length!"

    rounded_history = []
    for dt, seg_id, vehicle_id, start_offset, speed, status in sim.history.data:
        dt_rounded = round_datetime(dt, round_freq)
        rounded_history.append((dt_rounded, seg_id, vehicle_id, start_offset, speed, status))

    print("History rounded")
    aggregated_gv = GlobalView(data=rounded_history)
    unique_segments_in_time = set((row[0], row[1], segment_lengths[row[1]]) for row in aggregated_gv.data)

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


@click.command()
@click.argument("sim_path", type=click.Path(exists=True))
@click.option("--round_freq_s", type=int, default=300, help="How to round date times.")
@click.option("--border_id", type=str)
@click.option("--border_kind", type=str)
@click.option("--out", type=str, default="out.csv")
def aggregate_cmd(sim_path, round_freq_s, border_id, border_kind, out):
    return aggregate(sim_path, round_freq_s, border_id, border_kind, out)











