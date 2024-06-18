import logging
import time
from datetime import datetime, timedelta


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
