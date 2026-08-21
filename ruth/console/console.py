"""SimulationConsole: the end-of-step callback plus the command machinery.

Threading contract: the sim's main thread is the ONLY code that touches
simulation objects (inside end_step). The HTTP thread only reads the latest
pre-encoded snapshot blob (swapped atomically — a plain reference assignment
under the GIL) and enqueues validated commands.
"""

import logging
import queue
import threading
import time
from collections import deque
from datetime import timedelta

from .commands import Command, CommandResult
from .snapshot import SnapshotBuilder

logger = logging.getLogger(__name__)


class SimulationConsole:
    def __init__(self, console_args):
        self.args = console_args
        self.snapshot_builder = SnapshotBuilder(
            top_n=console_args.top_segments,
            max_vehicles=console_args.max_vehicles_in_snapshot,
        )

        self._snapshot_bytes = b'{"step": null, "waiting": true}'
        self._commands = queue.Queue()
        self._results = deque(maxlen=100)
        self._results_lock = threading.Lock()
        self._next_id = 1
        self._id_lock = threading.Lock()

        self._paused = False
        self._stop_requested = False
        self._step = 0
        self._last_step_end = None
        self._server = None

    # -- lifecycle (called from ruth.console.create_console) --------------

    def start(self):
        from .server import start_server
        self._server = start_server(self)
        logger.info("Console listening on http://%s:%d", self.args.host, self.args.port)

    # -- HTTP-thread API ---------------------------------------------------

    @property
    def snapshot_bytes(self):
        return self._snapshot_bytes

    def submit(self, command_type, payload=None):
        """Queue a validated command; returns its id (HTTP thread)."""
        with self._id_lock:
            cmd_id = self._next_id
            self._next_id += 1
        with self._results_lock:
            self._results.append(CommandResult(cmd_id, command_type, "pending"))
        self._commands.put(Command(cmd_id, command_type, payload or {}))
        return cmd_id

    def command_results(self):
        with self._results_lock:
            return [r.as_dict() for r in self._results]

    def geometry_for_ids(self, ids):
        return self.snapshot_builder.geometry_for_ids(ids)

    # -- sim-thread side ---------------------------------------------------

    def _set_result(self, cmd_id, status, detail=""):
        with self._results_lock:
            for r in self._results:
                if r.id == cmd_id:
                    r.status = status
                    r.detail = detail
                    return

    def _apply(self, sim, cmd):
        if cmd.type == "set-param":
            name, value = cmd.payload["name"], cmd.payload["value"]
            setting = sim.setting
            if name == "k-alternatives":
                setting.k_alternatives = int(value)
            elif name == "map-update-freq-s":
                setting.map_update_freq_s = timedelta(seconds=value)
            elif name == "los-vehicles-tolerance-s":
                setting.los_vehicles_tolerance = timedelta(seconds=value)
            elif name == "travel-time-limit-perc":
                setting.travel_time_limit_perc = float(value)
            elif name == "stuck-detection":
                setting.stuck_detection = int(value)
            else:
                raise ValueError(f"unknown parameter '{name}'")
            return f"{name} = {value}"

        if cmd.type == "restriction":
            from ruth.data.map import TemporarySpeed
            from ruth.data.segment import SpeedKph
            p = cmd.payload
            u, v = p["node_from"], p["node_to"]
            routing_map = sim.routing_map
            if not routing_map.current_network.has_edge(u, v):
                raise ValueError(f"no edge {u} -> {v} in the routing map")
            t_from = (sim.setting.departure_time + sim.compute_current_offset()
                      + timedelta(seconds=p["start_offset_s"]))
            t_to = t_from + timedelta(seconds=p["duration_s"])
            routing_map.temporary_speeds.append(TemporarySpeed(
                node_from=u,
                node_to=v,
                temporary_speed=SpeedKph(p["speed_kph"]),
                original_max_speed=SpeedKph(routing_map.get_original_max_speed(u, v)),
                timestamp_from=t_from,
                timestamp_to=t_to,
                active=False,
            ))
            # keep the init_temporary_max_speeds sort invariant
            routing_map.temporary_speeds.sort(key=lambda ts: ts.timestamp_from)
            return f"{u}->{v} @ {p['speed_kph']} km/h until {t_to}"

        if cmd.type == "pause":
            self._paused = True
            return "paused"
        if cmd.type == "resume":
            self._paused = False
            return "resumed"
        if cmd.type == "checkpoint":
            path = f"console-checkpoint-step{self._step}.pickle"
            sim.store(path)
            return path
        if cmd.type == "stop":
            self._stop_requested = True
            return "stop requested"
        raise ValueError(f"unknown command type '{cmd.type}'")

    def _drain_commands(self, sim):
        while True:
            try:
                cmd = self._commands.get_nowait()
            except queue.Empty:
                return
            try:
                detail = self._apply(sim, cmd)
                self._set_result(cmd.id, "ok", detail)
            except Exception as e:  # semantic failures must not kill the sim
                logger.warning("Console command %s failed: %s", cmd.type, e)
                self._set_result(cmd.id, "error", str(e))

    def _restrictions_view(self, sim):
        return [
            {
                "node_from": ts.node_from,
                "node_to": ts.node_to,
                "speed_kph": float(ts.temporary_speed),
                "from": ts.timestamp_from.isoformat(sep=" "),
                "to": ts.timestamp_to.isoformat(sep=" "),
                "active": ts.active,
            }
            for ts in sim.routing_map.temporary_speeds
        ]

    def end_step(self, sim):
        """The end_step_fns callback. Returns True to request a graceful stop."""
        self._step += 1
        now = time.perf_counter()
        last_step_ms = round((now - self._last_step_end) * 1000, 1) if self._last_step_end else None
        self._last_step_end = now

        self._drain_commands(sim)

        publish = (self.args.snapshot_every_steps <= 1
                   or self._step % self.args.snapshot_every_steps == 0)
        if publish or self._paused:
            self._snapshot_bytes = self.snapshot_builder.build(
                sim, self._step, self._paused, self._restrictions_view(sim), last_step_ms)

        # Pause: park the sim thread here, keep serving and draining commands.
        was_paused = self._paused
        while self._paused and not self._stop_requested:
            time.sleep(0.2)
            self._drain_commands(sim)
        if was_paused:
            # republish so the page stops showing the paused badge immediately
            self._snapshot_bytes = self.snapshot_builder.build(
                sim, self._step, self._paused, self._restrictions_view(sim), last_step_ms)

        return self._stop_requested
