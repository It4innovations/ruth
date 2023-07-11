import logging
from dataclasses import asdict
from datetime import timedelta
from itertools import product

import pandas as pd
from probduration import SegmentPosition

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
    return [Vehicle(**row.to_dict()) for (_, row) in df.iterrows()]


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
