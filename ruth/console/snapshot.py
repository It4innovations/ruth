"""Per-step JSON snapshot of the running simulation for the console.

Built inside the end-of-step callback (main thread) — the only writer of
simulation state — so no locking against the sim is needed. Everything shipped
to the HTTP thread is a plain pre-encoded bytes blob.
"""

import json
import math
from collections import Counter

import numpy as np


class SnapshotBuilder:
    PERF_HISTORY = 120  # rolling step-duration window shipped for the chart

    def __init__(self, top_n: int, max_vehicles: int):
        self.top_n = top_n
        self.max_vehicles = max_vehicles
        # rolling [step, duration_ms, need_new_route] rows
        self._perf_history = []
        # rolling per-phase timing series: {phase_name: [ms, ...]} aligned with
        # _perf_history (missing phases padded with None)
        self._phase_history = {}
        # (u, v) -> [[lat, lon], ...]; each segment's shapely work happens once.
        self._geometry_cache = {}
        # node id -> (lon, lat); built lazily on the first snapshot.
        self._node_xy = None
        # Immutable edge midpoint index for click-to-select (built once on the
        # sim thread; safe to read from the HTTP thread afterwards).
        self._edge_index = None

    # -- geometry helpers -------------------------------------------------

    def _ensure_node_xy(self, routing_map):
        if self._node_xy is None:
            self._node_xy = {
                node: (data.get("x"), data.get("y"))
                for node, data in routing_map.network.nodes(data=True)
            }
        return self._node_xy

    def segment_geometry(self, routing_map, node_from, node_to):
        """[[lat, lon], ...] polyline for a segment, memoized."""
        key = (node_from, node_to)
        if key not in self._geometry_cache:
            # Reuse flowmap's pure networkx/shapely polyline extraction
            # (honors curved edge 'geometry' when present).
            from ruth.flowmap.flowmapframe.plot import get_node_coordinates
            xs, ys = get_node_coordinates(routing_map.network, node_from, node_to)
            if xs is None:
                self._geometry_cache[key] = None
            else:
                self._geometry_cache[key] = [[lat, lon] for lon, lat in zip(xs, ys)]
        return self._geometry_cache[key]

    def _ensure_edge_index(self, routing_map):
        if self._edge_index is None:
            node_xy = self._ensure_node_xy(routing_map)
            edges, mids = [], []
            for u, v in routing_map.current_network.edges():
                xu, xv = node_xy.get(u), node_xy.get(v)
                if not xu or not xv or xu[0] is None or xv[0] is None:
                    continue
                edges.append((u, v))
                # midpoint plus endpoints, all as (lat, lon)
                mids.append(((xu[1] + xv[1]) / 2.0, (xu[0] + xv[0]) / 2.0,
                             xu[1], xu[0], xv[1], xv[0]))
            self._edge_index = (edges, np.asarray(mids, dtype=np.float64))
        return self._edge_index

    def nearest_segment(self, lat, lon):
        """Nearest edge to a clicked point. Thread-safe once the index exists."""
        if self._edge_index is None:
            return None
        edges, mids = self._edge_index
        if not edges:
            return None
        # squared equirectangular distance; lon scaled by cos(lat)
        scale = math.cos(math.radians(lat))
        d2 = (mids[:, 0] - lat) ** 2 + ((mids[:, 1] - lon) * scale) ** 2
        i = int(np.argmin(d2))
        u, v = edges[i]
        # memoized curved geometry when available, straight endpoints otherwise
        geometry = self._geometry_cache.get((u, v)) or [
            [mids[i, 2], mids[i, 3]], [mids[i, 4], mids[i, 5]]]
        return {"node_from": u, "node_to": v, "id": f"{u}-{v}", "geometry": geometry}

    def geometry_for_ids(self, ids):
        """Serve already-memoized polylines for 'U-V' id strings."""
        out = {}
        for sid in ids:
            try:
                u, v = (int(p) for p in sid.split("-"))
            except ValueError:
                continue
            geom = self._geometry_cache.get((u, v))
            if geom:
                out[sid] = geom
        return out

    # -- snapshot ---------------------------------------------------------

    def _vehicle_position(self, vehicle, routing_map, node_xy):
        """Interpolated (lat, lon) along the vehicle's current segment."""
        route = vehicle.osm_route
        i = vehicle.start_index
        if i >= len(route):
            return None
        u = route[i]
        xy_u = node_xy.get(u)
        if xy_u is None or xy_u[0] is None:
            return None
        if i + 1 >= len(route):
            return (xy_u[1], xy_u[0])
        v = route[i + 1]
        xy_v = node_xy.get(v)
        if xy_v is None or xy_v[0] is None:
            return (xy_u[1], xy_u[0])
        try:
            length = routing_map.current_network[u][v]["length"]
            t = min(1.0, max(0.0, float(vehicle.start_distance_offset) / float(length))) if length else 0.0
        except KeyError:
            t = 0.0
        lon = xy_u[0] + (xy_v[0] - xy_u[0]) * t
        lat = xy_u[1] + (xy_v[1] - xy_u[1]) * t
        return (lat, lon)

    def build(self, sim, step, paused, restrictions_view, last_step_ms):
        """Return the snapshot as a JSON-encoded bytes blob."""
        routing_map = sim.routing_map
        node_xy = self._ensure_node_xy(routing_map)
        self._ensure_edge_index(routing_map)

        active = [v for v in sim.vehicles if v.active]

        # Segments with the most traffic at the CURRENT step: exact count over
        # every active vehicle's current segment (same identity FCDRecord uses).
        counts = Counter(
            (v.osm_route[v.start_index], v.osm_route[v.start_index + 1])
            for v in active
            if v.start_index + 1 < len(v.osm_route)
        )
        top_segments = []
        for (u, v), count in counts.most_common(self.top_n):
            entry = {"id": f"{u}-{v}", "count": count}
            try:
                edge = routing_map.current_network[u][v]
                entry["current_speed_kph"] = round(float(edge.get("current_speed", edge.get("speed_kph", 0.0))), 1)
                entry["max_speed_kph"] = round(float(edge.get("speed_kph", 0.0)), 1)
            except KeyError:
                pass
            entry["geometry"] = self.segment_geometry(routing_map, u, v)
            top_segments.append(entry)

        # Vehicle dot cloud, stride-sampled above the cap (top-N stays exact).
        stride = 1
        if len(active) > self.max_vehicles:
            stride = math.ceil(len(active) / self.max_vehicles)
        vehicles = []
        for vehicle in active[::stride]:
            pos = self._vehicle_position(vehicle, routing_map, node_xy)
            if pos is not None:
                # dest_node lets the page color vehicles by destination
                vehicles.append([round(pos[0], 6), round(pos[1], 6),
                                 int(getattr(vehicle, "dest_node", 0) or 0)])

        # Simulation-domain metrics from the segment counts (cheap: O(unique
        # occupied segments), weights = vehicles on the segment).
        total_w = 0
        speed_w = 0.0
        ratio_w = 0.0
        for (u, v), count in counts.items():
            try:
                edge = routing_map.current_network[u][v]
                cur = float(edge.get("current_speed", edge.get("speed_kph", 0.0)))
                mx = float(edge.get("speed_kph", 0.0)) or 1.0
            except KeyError:
                continue
            total_w += count
            speed_w += cur * count
            ratio_w += min(1.0, cur / mx) * count
        mean_speed = round(speed_w / total_w, 1) if total_w else None
        congestion = round(1.0 - ratio_w / total_w, 3) if total_w else None

        # Performance metrics: singlenode stores a StepInfo (duration + phase
        # timings + reroute count) for the just-finished step BEFORE the
        # end-of-step callbacks run, so steps_info[-1] is current.
        perf = None
        records_seen = None
        steps_info = getattr(sim, "steps_info", None)
        if steps_info:
            si = steps_info[-1]
            duration_ms = round(si.duration.total_seconds() * 1000, 1)
            records_seen = int(si.parts.get("fcd_records_seen", 0))
            new_row = not self._perf_history or self._perf_history[-1][0] != si.step
            if new_row:
                # one row per step carrying every charted series:
                # [step, duration_ms, rerouting, n_active, mean_speed,
                #  congestion, occupied_segments, records_seen]
                self._perf_history.append([si.step, duration_ms, si.need_new_route,
                                           len(active), mean_speed, congestion,
                                           len(counts), records_seen])
                if len(self._perf_history) > self.PERF_HISTORY:
                    del self._perf_history[:-self.PERF_HISTORY]
            parts = {k: round(float(v), 2) for k, v in si.parts.items()
                     if not k.startswith("fcd_records") and float(v) >= 0.005}
            if new_row:
                depth = len(self._perf_history)
                for k in set(self._phase_history) | set(parts):
                    series = self._phase_history.setdefault(k, [None] * (depth - 1))
                    series.append(parts.get(k))
                    if len(series) > self.PERF_HISTORY:
                        del series[:-self.PERF_HISTORY]
            perf = {
                "duration_ms": duration_ms,
                "need_new_route": si.need_new_route,
                "parts_ms": dict(sorted(parts.items(), key=lambda kv: -kv[1])),
                "history": self._perf_history,
                "phases_history": self._phase_history,
            }

        sim_metrics = {
            "occupied_segments": len(counts),
            "mean_speed_kph": mean_speed,
            "congestion_index": congestion,
            "rerouting": perf["need_new_route"] if perf else None,
            "records_seen": records_seen,
            "sim_speedup": (round(getattr(sim.setting, "round_freq", None).total_seconds() * 1000.0
                                  / perf["duration_ms"], 1)
                            if perf and perf["duration_ms"] and getattr(sim.setting, "round_freq", None)
                            else None),
        }

        setting = sim.setting
        snapshot = {
            "step": step,
            "offset_s": sim.compute_current_offset().total_seconds() if active else None,
            "paused": paused,
            "n_total": len(sim.vehicles),
            "n_active": len(active),
            "n_finished": len(sim.vehicles) - len(active),
            "settings": {
                "k-alternatives": setting.k_alternatives,
                "map-update-freq-s": setting.map_update_freq_s.total_seconds(),
                "los-vehicles-tolerance-s": setting.los_vehicles_tolerance.total_seconds(),
                "travel-time-limit-perc": setting.travel_time_limit_perc,
                "stuck-detection": setting.stuck_detection,
            },
            "top_segments": top_segments,
            "restrictions": restrictions_view,
            "vehicles": vehicles,
            "vehicles_sampled": stride > 1,
            "sample_stride": stride,
            "last_step_ms": last_step_ms,
            "perf": perf,
            "metrics": sim_metrics,
        }
        return json.dumps(snapshot).encode("utf-8")
