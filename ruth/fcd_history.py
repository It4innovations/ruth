import glob
import logging
import os
import queue
import re
import time
import traceback
from datetime import datetime
from multiprocessing import get_context
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np

from .data.hdf_stream_writer import HDF5Writer, fcd_records_to_array

if TYPE_CHECKING:
    from .simulator.simulation import FCDRecord

logger = logging.getLogger(__name__)


def _part_path(base_path: str, part: int) -> str:
    base_no_ext = os.path.splitext(base_path)[0]
    return f"{base_no_ext}-part{part:04d}.h5"


def _existing_part_numbers(base_path: str) -> List[int]:
    base_no_ext = os.path.splitext(base_path)[0]
    paths = glob.glob(f"{base_no_ext}-part*.h5")
    part_nums = []
    for p in paths:
        m = re.search(r"-part(\d+)\.h5$", os.path.basename(p))
        if m:
            try:
                part_nums.append(int(m.group(1)))
            except Exception:
                pass
    return part_nums


def _map_metadata(routing_map, departure_time, round_freq) -> Dict:
    return {
        "bbox": tuple(routing_map.bbox.get_coords()),
        "download_date": str(routing_map.download_date),
        "departure_time_iso": departure_time.isoformat(),
        "round_freq_s": float(round_freq.total_seconds()),
    }


def _writer_process_main(base_path: str, max_records_per_file: int, first_part: int, q, stats_q):
    """Single HDF5 writer process.

    The writer owns the HDF5 file handle. The simulator process sends metadata,
    computational-time updates, and NumPy arrays. File rotation remains here, so
    the simulator process does not need to know the current HDF5 part index.
    """
    writer: Optional[HDF5Writer] = None
    current_part = first_part
    records_written = 0
    batches_written = 0
    write_seconds = 0.0
    opened_files = []

    def ensure_writer() -> HDF5Writer:
        nonlocal writer, current_part
        if writer is None:
            path = _part_path(base_path, current_part)
            writer = HDF5Writer(path)
            opened_files.append(path)
        return writer

    def rotate_writer():
        nonlocal writer, current_part
        if writer is not None:
            writer.close()
        writer = None
        current_part += 1
        ensure_writer()

    def append_array_rotating(data: np.ndarray):
        nonlocal records_written, batches_written, write_seconds
        if data is None or len(data) == 0:
            return

        start = 0
        n = len(data)
        while start < n:
            w = ensure_writer()
            remaining_in_file = max_records_per_file - w.index
            if remaining_in_file <= 0:
                rotate_writer()
                continue

            take = min(n - start, remaining_in_file)
            chunk = data[start:start + take]
            t0 = time.perf_counter()
            w.append_array(chunk)
            write_seconds += time.perf_counter() - t0
            records_written += len(chunk)
            batches_written += 1
            start += take

    try:
        while True:
            msg_type, payload = q.get()

            if msg_type == "array":
                append_array_rotating(payload)

            elif msg_type == "map":
                w = ensure_writer()
                w.save_map_metadata(**payload)

            elif msg_type == "computational_time":
                w = ensure_writer()
                w.save_computational_time(float(payload))

            elif msg_type == "close":
                break

            else:
                raise ValueError(f"Unknown FCD writer message type: {msg_type}")

        if writer is not None:
            writer.close()
            writer = None

        stats_q.put({
            "records_written": records_written,
            "batches_written": batches_written,
            "write_seconds": write_seconds,
            "opened_files": opened_files,
            "error": None,
        })

    except BaseException:
        err = traceback.format_exc()
        try:
            stats_q.put({
                "records_written": records_written,
                "batches_written": batches_written,
                "write_seconds": write_seconds,
                "opened_files": opened_files,
                "error": err,
            })
        finally:
            if writer is not None:
                writer.close()
        raise


class _WriterFacade:
    """Backward-compatible facade used by existing code as history.writer."""

    def __init__(self, history: "FCDHistory"):
        self.history = history

    def save_map(self, routing_map, departure_time, round_freq):
        self.history.save_map(routing_map, departure_time, round_freq)

    def save_computational_time(self, computational_time: float):
        self.history.save_computational_time(computational_time)

    def close(self):
        self.history.close()


class FCDHistory:
    def __init__(
            self,
            h5_path_base: str,
            buffer_size: int,
            max_records_per_file: int,
            async_enabled: bool = False,
            queue_size: int = 4,
    ):
        self.base_path = h5_path_base
        self.buffer_size = int(buffer_size)
        self.buffer: List["FCDRecord"] = []
        self.fcd_history: List["FCDRecord"] = []  # kept for pickle compatibility; no longer populated
        self.start_time = None
        self.max_records_per_file = int(max_records_per_file)
        self._current_part = 0

        self.async_enabled = bool(async_enabled)
        self.queue_size = int(queue_size)

        self.writer = None              # facade exposed for existing code
        self._sync_writer = None         # HDF5Writer in sync mode only
        self._ctx = None
        self._queue = None
        self._stats_queue = None
        self._process = None
        self._closed = False

        self._recent_metrics: Dict[str, float] = {}
        self._total_metrics: Dict[str, float] = {}

    def __enter__(self):
        self._ensure_started()
        self.start_time = datetime.now()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Preserve previous behavior: save wall-clock computation time if the
        # simulation loop did not already write simulation.duration.
        if self.start_time:
            computational_time = (datetime.now() - self.start_time).total_seconds()
            self.save_computational_time(computational_time)
        self.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in ["writer", "_sync_writer", "_ctx", "_queue", "_stats_queue", "_process"]:
            state.pop(key, None)
        return state

    def __setstate__(self, state):
        if not isinstance(state, dict):
            raise TypeError(f"Expected dict for state, got {type(state)}")

        self.__dict__.update(state)
        self.writer = None
        self._sync_writer = None
        self._ctx = None
        self._queue = None
        self._stats_queue = None
        self._process = None
        self._closed = False

        if not hasattr(self, 'base_path'):
            self.base_path = self.path
        if not hasattr(self, "async_enabled"):
            self.async_enabled = False
        if not hasattr(self, "queue_size"):
            self.queue_size = 4
        if not hasattr(self, "_recent_metrics"):
            self._recent_metrics = {}
        if not hasattr(self, "_total_metrics"):
            self._total_metrics = {}
        if not hasattr(self, "_current_part"):
            self._current_part = 0
        else:
            self._current_part += 1

        existing = _existing_part_numbers(self.base_path)
        if existing:
            max_part = max(existing)
            if max_part >= self._current_part:
                logging.warning(
                    "FCD history part files exist for base '%s'. Continuing from part %s.",
                    self.base_path,
                    max_part + 1,
                )
                self._current_part = max_part + 1

    def _add_metric(self, name: str, value: float):
        self._recent_metrics[name] = self._recent_metrics.get(name, 0.0) + float(value)
        self._total_metrics[name] = self._total_metrics.get(name, 0.0) + float(value)

    def collect_step_metrics(self, reset: bool = True) -> Dict[str, float]:
        metrics = dict(self._recent_metrics)
        if reset:
            self._recent_metrics.clear()
        return metrics

    def collect_total_metrics(self) -> Dict[str, float]:
        return dict(self._total_metrics)

    def _ensure_started(self):
        if self.writer is not None:
            return

        self.writer = _WriterFacade(self)

        if self.async_enabled:
            self._ctx = get_context("spawn")
            self._queue = self._ctx.Queue(maxsize=self.queue_size)
            self._stats_queue = self._ctx.Queue(maxsize=1)
            self._process = self._ctx.Process(
                target=_writer_process_main,
                args=(self.base_path, self.max_records_per_file, self._current_part,
                      self._queue, self._stats_queue),
                name="ruth-fcd-hdf5-writer",
            )
            self._process.daemon = False
            self._process.start()
            logger.info(
                "Started async FCD HDF5 writer process pid=%s queue_size=%s buffer_size=%s",
                self._process.pid,
                self.queue_size,
                self.buffer_size,
            )
        else:
            self._open_existing_writer()
            logger.info("Using synchronous FCD HDF5 writer buffer_size=%s", self.buffer_size)

    def _open_existing_writer(self):
        path = _part_path(self.base_path, self._current_part)
        self.path = path
        self._sync_writer = HDF5Writer(path)

    def _rotate_sync_writer(self):
        if self._sync_writer is not None:
            self._sync_writer.close()
        self._current_part += 1
        self._open_existing_writer()

    def _append_array_sync_rotating(self, data: np.ndarray):
        start = 0
        n = len(data)
        while start < n:
            if self._sync_writer is None:
                self._open_existing_writer()

            remaining_in_file = self.max_records_per_file - self._sync_writer.index
            if remaining_in_file <= 0:
                self._rotate_sync_writer()
                continue

            take = min(n - start, remaining_in_file)
            chunk = data[start:start + take]
            t0 = time.perf_counter()
            self._sync_writer.append_array(chunk)
            self._add_metric("fcd_write_ms", (time.perf_counter() - t0) * 1000.0)
            start += take

    def _put_async(self, msg_type: str, payload):
        self._ensure_started()
        t0 = time.perf_counter()
        self._queue.put((msg_type, payload), block=True)
        enqueue_ms = (time.perf_counter() - t0) * 1000.0
        self._add_metric("fcd_enqueue_ms", enqueue_ms)
        if enqueue_ms > 100.0:
            self._add_metric("fcd_queue_wait_events", 1.0)
        return enqueue_ms

    def save_map(self, routing_map, departure_time, round_freq):
        self._ensure_started()
        metadata = _map_metadata(routing_map, departure_time, round_freq)
        if self.async_enabled:
            self._put_async("map", metadata)
        else:
            self._sync_writer.save_map_metadata(**metadata)

    def save_computational_time(self, computational_time: float):
        self._ensure_started()
        if self.async_enabled:
            self._put_async("computational_time", float(computational_time))
        else:
            self._sync_writer.save_computational_time(float(computational_time))

    def extend(self, fcd: List["FCDRecord"]):
        if not fcd:
            return

        self._add_metric("fcd_records_seen", len(fcd))
        self.buffer.extend(fcd)

        if len(self.buffer) >= self.buffer_size:
            self.flush_to_disk()

    def flush_to_disk(self):
        if not self.buffer:
            return

        self._ensure_started()
        n_records = len(self.buffer)

        t0 = time.perf_counter()
        data = fcd_records_to_array(self.buffer)
        self._add_metric("fcd_convert_ms", (time.perf_counter() - t0) * 1000.0)

        # Release FCDRecord object references as early as possible. The async
        # writer receives the compact structured array, not Vehicle/Segment objects.
        self.buffer.clear()

        self._add_metric("fcd_flush_batches", 1.0)
        self._add_metric("fcd_records_flushed", n_records)

        if self.async_enabled:
            self._put_async("array", data)
        else:
            self._append_array_sync_rotating(data)

    def close(self):
        if self._closed:
            return

        self.flush_to_disk()

        if self.async_enabled:
            if self._queue is not None:
                self._put_async("close", None)

            if self._process is not None:
                self._process.join()

            stats = None
            if self._stats_queue is not None:
                try:
                    stats = self._stats_queue.get_nowait()
                except queue.Empty:
                    stats = None

            if stats:
                logger.info(
                    "Async FCD writer finished: records=%s batches=%s write_time_s=%.3f files=%s",
                    stats.get("records_written"),
                    stats.get("batches_written"),
                    float(stats.get("write_seconds", 0.0)),
                    stats.get("opened_files"),
                )
                if stats.get("error"):
                    raise RuntimeError(f"Async FCD writer failed:\n{stats['error']}")

            if self._process is not None and self._process.exitcode not in (0, None):
                raise RuntimeError(f"Async FCD writer exited with code {self._process.exitcode}")

        else:
            if self._sync_writer is not None:
                self._sync_writer.close()

        self._closed = True

    def to_dataframe(self):
        logging.warning("This function will be deprecated soon with migration to h5 storage for FCD history.")
        raise NotImplementedError("to_dataframe is disabled when streaming to HDF5.")
