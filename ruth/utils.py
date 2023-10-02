import logging
import re
import time
from datetime import datetime, timedelta

from .data.border import Border, BorderType, PolygonBorderDef
from .data.map import Map
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
    name_ = name if name is not None else f"custom_{border_def.md5()}_{border_kind.name.lower()}"
    border = Border(name_, border_def, border_kind, data_dir, load_from_cache)

    return Map(border, with_speeds=with_speeds)


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


def is_root_debug_logging() -> bool:
    """
    Returns true if the global (root) logger has at least `logging.DEBUG` level.
    """
    return logging.getLogger().isEnabledFor(logging.DEBUG)


class Timer:

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        self.end = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()

    @property
    def duration_ms(self):
        assert self.end is not None, "Trying to call duration on unfinished timer."
        return (self.end - self.start) * 1000


class TimerSet:

    def __init__(self):
        self.timers = []

    def get(self, name):
        self.timers.append(Timer(name))
        return self.timers[-1]

    def collect(self):
        return dict((timer.name, timer.duration_ms) for timer in self.timers)
