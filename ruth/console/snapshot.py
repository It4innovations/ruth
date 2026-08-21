"""Per-step JSON snapshot of the running simulation for the console.

Built inside the end-of-step callback (main thread) — the only writer of
simulation state — so no locking against the sim is needed. Everything shipped
to the HTTP thread is a plain pre-encoded bytes blob.
"""

import json
import math
from collections import Counter


class SnapshotBuilder:
    def __init__(self, top_n: int, max_vehicles: int):
        self.top_n = top_n
        self.max_vehicles = max_vehicles
        # (u, v) -> [[lat, lon], ...]; each segment's shapely work happens once.
        self._geometry_cache = {}
        # node id -> (lon, lat); built lazily on the first snapshot.
        self._node_xy = None

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
                vehicles.append([round(pos[0], 6), round(pos[1], 6), str(vehicle.status)])

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
        }
        return json.dumps(snapshot).encode("utf-8")
