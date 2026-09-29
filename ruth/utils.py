import gc
import logging
import time
from datetime import datetime, timedelta, timezone


def datetime_to_timestamp(dt) -> float:
    return dt.replace(tzinfo=timezone.utc).timestamp()


def round_timedelta(td: timedelta, freq: timedelta):
    td_seconds = td.total_seconds()
    freq_seconds = freq.total_seconds()
    return timedelta(seconds=freq_seconds * round(td_seconds / freq_seconds))


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

    def __init__(self, name: str, profile: bool = False):
        self.name = name
        self.profile = profile

    def __enter__(self):
        self.end = None
        if self.profile:
            self.gc_ms = 0.0
            self.gc_collections = [0, 0, 0]
            self._gc_started = {}
            self._gc_callback = self._record_gc
            gc.callbacks.append(self._gc_callback)
            self.cpu_start = time.process_time()
        self.start = time.perf_counter()
        return self

    def _record_gc(self, phase, info):
        generation = info["generation"]
        if phase == "start":
            self._gc_started[generation] = time.perf_counter()
        elif phase == "stop":
            started = self._gc_started.pop(generation, None)
            if started is not None:
                self.gc_ms += (time.perf_counter() - started) * 1000.0
                self.gc_collections[generation] += 1

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.perf_counter()
        if self.profile:
            self.cpu_end = time.process_time()
            gc.callbacks.remove(self._gc_callback)
            del self._gc_callback  # Do not leave a bound-method reference cycle.

    @property
    def profile_metrics(self):
        """Process CPU includes all its threads; GC time overlaps phase wall time.

        Scoped callbacks observe collections without forcing or disabling GC.
        Nested timers may attribute the same collection to both phases.
        """
        if not self.profile:
            return {}
        assert self.end is not None, "Trying to collect an unfinished timer."
        metrics = {
            f"{self.name}_cpu_ms": (self.cpu_end - self.cpu_start) * 1000.0,
            f"{self.name}_gc_ms": self.gc_ms,
        }
        metrics.update({
            f"{self.name}_gc_gen{generation}_collections": count
            for generation, count in enumerate(self.gc_collections)
        })
        return metrics

    @property
    def duration_ms(self):
        assert self.end is not None, "Trying to call duration on unfinished timer."
        return (self.end - self.start) * 1000


class TimerSet:

    def __init__(self):
        self.timers = []

    def get(self, name, profile=False):
        self.timers.append(Timer(name, profile=profile))
        return self.timers[-1]

    def collect(self):
        metrics = {}
        for timer in self.timers:
            metrics[timer.name] = timer.duration_ms
            metrics.update(timer.profile_metrics)
        return metrics
