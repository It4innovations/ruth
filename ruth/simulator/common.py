import logging
from dataclasses import asdict
from datetime import timedelta
from itertools import product

import pandas as pd
from probduration import SegmentPosition

from ..data.map import BBox
from ..vehicle import Vehicle

logger = logging.getLogger(__name__)


class ParamsGenerator:
    def __init__(self):
        self.params = dict()

    def register(self, name, iterable):
        assert name not in self.params, f"Parameter '{name}' is already registered."
        self.params[name] = iterable

    def drop(self, name):
        del self.params[name]

    def __iter__(self):
        prod = product(*self.params.values())
        keys = self.params.keys()

        for comb in prod:
            yield dict(zip(keys, comb))

    def __len__(self):
        return len(list(product(*self.params.values())))


def load_vehicles(input_path: str):
    logger.info("Loading data ... %s", input_path)
    df = pd.read_parquet(input_path, engine="fastparquet")
    vehicles = [Vehicle(
        id=row["id"],
        time_offset=row["time_offset"],
        frequency=row["frequency"],
        start_index=row["start_index"],
        start_distance_offset=row["start_distance_offset"],
        origin_node=row["origin_node"],
        dest_node=row["dest_node"],
        osm_route=row["osm_route"],
        active=row["active"],
        fcd_sampling_period=row["fcd_sampling_period"],
        status=row["status"],

    ) for (_, row) in df.iterrows()]

    bbox_lat_max = df["bbox_lat_max"].iloc[0]
    bbox_lon_min = df["bbox_lon_min"].iloc[0]
    bbox_lat_min = df["bbox_lat_min"].iloc[0]
    bbox_lon_max = df["bbox_lon_max"].iloc[0]
    download_date = df["download_date"].iloc[0]
    bbox = BBox(bbox_lat_max, bbox_lon_min, bbox_lat_min, bbox_lon_max)
    return vehicles, bbox, download_date


def save_vehicles(vehicles, output_path: str):
    logger.info("Saving vehicles ... %s", output_path)

    df = pd.DataFrame([asdict(v) for v in vehicles])
    df.to_pickle(output_path)


def distance_duration(driving_route, departure_time, los_db, rnd_gen, stop_distance=None):
    distance = 0.0
    p = SegmentPosition(0, 0.0)
    dt = departure_time
    level_of_services = []

    stop_distance_ = stop_distance if stop_distance is not None else driving_route.distance_in_meters()

    while distance < stop_distance_:
        if p.index >= len(driving_route):
            break

        seg = driving_route[p.index]
        los = los_db.get(dt, seg, rnd_gen())

        if los == float('inf'):  # stuck in traffic jam; not moving
            return float('inf'), None, None

        elapsed, next_segment_pos, assigned_speed_mps = driving_route.advance(p, dt, los)
        d = elapsed - dt

        if p.index == next_segment_pos.index:
            # movement on the same segment
            distance += next_segment_pos.position - p.start
        else:
            # if the next segment is different, i.e. its index is higher than
            # the rest distance of the previous segment is added.
            distance += seg.length - p.start

        if distance > stop_distance_:
            # round down to the stop distance

            # decrease the distance
            dd = distance - stop_distance_
            if next_segment_pos.position - dd < 0:
                next_segment_pos = SegmentPosition(p.index,
                                                   seg.length + (next_segment_pos.position - dd))
            else:
                next_segment_pos.position -= dd

            # decrease the duration by overpass time
            over_duration = dd / assigned_speed_mps
            d -= timedelta(seconds=over_duration)

        level_of_services.append(los)
        dt += d
        p = next_segment_pos

    # returns:
    # duration, next segment position, average level of service
    if len(level_of_services) == 0:
        return float("inf"), None, None

    avg_los = sum(level_of_services) / len(level_of_services)

    return dt - departure_time, p, avg_los
