import re
import osmnx as ox
from probduration import Segment
from datetime import datetime, timedelta
from contextlib import contextmanager
import time

from .data.map import Map
from .data.border import Border, BorderType, PolygonBorderDef
from .metaclasses import Singleton


def get_map(polygon: str,
            kind: BorderType,
            name=None,
            on_disk=False,
            with_speeds=False,
            data_dir="./data",
            load_from_cache=True):

    """Get map based on polygon."""
    border_def = PolygonBorderDef(polygon, on_disk=on_disk)
    border_kind = BorderType.parse(kind)
    name_ = name if name is not None else f"custom_{border_def.md5()}"
    border = Border(name_, border_def, border_kind, data_dir, load_from_cache)

    return Map(border, with_speeds=with_speeds)


def osm_route_to_segments(osm_route, routing_map):
    """Prepare list of segments based on route."""
    edge_data = ox.utils_graph.get_route_edge_attributes(routing_map.network,
                                                         osm_route)
    edges = zip(osm_route, osm_route[1:])
    return [
        Segment(
            f"OSM{from_}T{to_}",
            data["length"],
            data["speed_kph"],
        )
        # NOTE: the zip is correct as the starting node_id is of the interest
        for i, ((from_, to_), data) in enumerate(zip(edges, edge_data))
    ]


class SegmentIdParser:

    def __init__(self, metaclass=Singleton):
        self.osm_id_regex = re.compile("OSM(?P<node_from>\d+)T(?P<node_to>\d+)")

    def parse(self, segment_id):
        res = self.osm_id_regex.match(segment_id)
        assert res is not None, \
            "Invalid format of segment ID. It is exected the format: 'OSM<node_from>T<node_to>'"
        node_from, node_to = res.groups()
        return (int(node_from), int(node_to))


def parse_segment_id(segment_id):
    p = SegmentIdParser()

    return p.parse(segment_id)


def route_to_osm_route(route):
    osm_route = []
    for i in range(len(route) - 1):
        seg = route[i]
        node_from, _ = parse_segment_id(seg.id)
        osm_route.append(node_from)
    last_seg = route[-1]
    node_from, node_to = parse_segment_id(last_seg.id)
    osm_route.extend([node_from, node_to])

    return osm_route


def round_timedelta(td: timedelta, freq: timedelta):
    return freq * round(td / freq)


def round_datetime(dt: datetime, freq: timedelta):
    if freq / timedelta(hours=1) > 1:
        assert False, "Too rough rounding frequency"
    elif freq / timedelta(minutes=1) > 1:
        td = timedelta(minutes=dt.minute, seconds=dt.second, microseconds=dt.microsecond)
    elif freq / timedelta(seconds=1) > 1:
        td = timedelta(seconds=dt.second, microseconds=dt.microsecond)
    else:
        assert False, "Too fine rounding frequency"

    rest = dt - td
    td_rounded = round_timedelta(td, freq)

    return rest + td_rounded


@contextmanager
def timer(name: str, enabled=False):
    if enabled:
        start = time.time()
        try:
            yield
        finally:
            end = time.time()
            duration = end - start
            duration_ms = duration * 1000
            print(f"{name}: {duration_ms:.3f} ms")
    else:
        yield
