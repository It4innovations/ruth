"""Routing map module."""
import csv
import itertools
import os
import pickle
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from time import time

import networkx as nx
import osmnx as ox
import osmnx.settings
from networkx import MultiDiGraph
from osmnx import load_graphml, save_graphml, graph_from_bbox
from networkx.exception import NetworkXNoPath

from .hdf5_writer import get_edge_id_from_data, save_graph_to_hdf5
from ..log import console_logger as cl
from ..data.segment import Route, Segment, SegmentId, SpeedKph


@dataclass
class TemporarySpeed:
    node_from: int
    node_to: int
    temporary_speed: SpeedKph
    original_max_speed: SpeedKph
    timestamp_from: datetime
    timestamp_to: datetime
    active: bool


def segment_weight(n1, n2, data):
    return float(data['length']) + float(f"0.{n1}{n2}")


def save(G, fname):
    nx.write_gml(G, fname)


def get_osm_segment_id(node_from: int, node_to: int):
    return f"OSM{node_from}T{node_to}"


def round_speed(speed: float) -> int:
    return int(speed + 0.5)


def osm_route_to_segment_ids(route: Route) -> List[str]:
    """Turn routes made of osm node_ids to osm numerical segment ids."""
    return [get_osm_segment_id(node_from, node_to) for node_from, node_to in zip(route[:-1], route[1:])]


def osm_routes_to_segment_ids(routes: List[Route]) -> List[List[str]]:
    """Turn routes made of osm node_ids to osm numerical segment ids."""
    return [osm_route_to_segment_ids(route) for route in routes]


highway_types = ["motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link", "secondary",
                 "secondary_link", "tertiary", "tertiary_link", "unclassified", "residential", "living_street",
                 "service", "track", "road", "path", "footway", "steps", "cycleway", "bridleway", "pedestrian",
                 "construction", "proposed", "bus_guideway", "escape", "raceway", "services", "rest_area", "platform",
                 "crossing", "elevator", "corridor", "abandoned", "planned", "no", "disused", "razed", "historic:",
                 "busway"]

highway_dict = {highway: i for i, highway in enumerate(highway_types)}


class BBox:
    def __init__(self, north, west, south, east):
        self.north = north
        self.west = west
        self.south = south
        self.east = east

    def get_coords(self):
        return self.north, self.west, self.south, self.east

    @property
    def name(self):
        return f"{self.north}-{self.west}-{self.south}-{self.east}".replace('.', "_")


class Map:
    """Routing map."""

    def __init__(self, bbox, download_date, data_dir="./data", with_speeds=True, save_hdf=True):
        """
        Initialize a map via the border.
        If `data_dir` provided, the map is loaded from locally stored maps preferably.
        """
        self.bbox = bbox
        self.download_date = download_date
        self.data_dir = data_dir
        self.network, fresh_data = self._load()
        self.temporary_speeds = []

        if fresh_data:
            for u, v, k, data in self.network.edges(keys=True, data=True):
                if 'highway' in data and type(data['highway']) is list:
                    self.network[u][v][k]['highway'] = sorted(data['highway'], key=lambda x: highway_dict[x])

        if with_speeds:
            self.network = ox.add_edge_speeds(self.network)

        self.original_network = ox.get_digraph(self.network)

        self.segment_lengths = {}
        for i, (node_from, node_to) in enumerate(self.original_network.edges()):
            edge = self.original_network[node_from][node_to]
            edge['routing_id'] = i + 1
            edge['length'] = int(edge['length'] + 0.5)
            edge['speed_kph'] = round_speed(edge['speed_kph'])
            edge['maxspeed'] = edge['speed_kph']
            self.segment_lengths[(node_from, node_to)] = edge['length']

        self.current_network = self.original_network.copy()

        if with_speeds:
            self.init_current_speeds()

        if fresh_data:
            self._store()

        self.network = MultiDiGraph(self.original_network)

        if save_hdf:
            hdf5_file_name = f"map_{round(time() * 1000)}_{os.getpid()}.hdf5"
            self.hdf5_file_path = str((Path(self.data_dir) / hdf5_file_name).absolute())
            self.save_hdf()

    def save_hdf(self) -> str:
        temp_path = str((Path(self.data_dir) / f"map_{round(time() * 1000)}_{os.getpid()}_temp.hdf5").absolute())

        self.osm_to_hdf_map_ids = save_graph_to_hdf5(self.current_network, temp_path)
        self.hdf_to_osm_map_ids = {v: k for (k, v) in self.osm_to_hdf_map_ids.items()}

        assert len(self.osm_to_hdf_map_ids) == len(self.hdf_to_osm_map_ids)
        shutil.move(temp_path, self.hdf5_file_path)
        return self.hdf5_file_path

    @staticmethod
    def from_memory(pickle_state):
        m = pickle.loads(pickle_state)
        m.data_dir = None
        return m

    @property
    def file_path(self):
        if self.data_dir is None:
            return None
        """Path to locally stored map."""
        return os.path.join(self.data_dir, f"{self.name}.graphml")

    @property
    def name(self):
        """Name of the map."""
        return self.bbox.name + "_" + self.download_date.replace(":", "-")

    def edges(self):
        return self.current_network.edges()

    def get_travel_time(self, node_from: int, node_to: int, speed_kph: SpeedKph) -> float:
        if speed_kph == 0:
            return float('inf')
        segment_length_m = float(self.segment_lengths[(node_from, node_to)])
        speed_mps = float(speed_kph) / 3.6
        travel_time_s = segment_length_m / speed_mps
        return travel_time_s

    def get_path_travel_time(self, path: List[int]):
        total_time = 0
        for node_from, node_to in zip(path[:-1], path[1:]):
            total_time += self.current_network[node_from][node_to]['current_travel_time']
        return total_time

    def check_if_travel_time_is_faster(self, path: List[int], time_limit: float):
        total_time = 0
        for node_from, node_to in zip(path[:-1], path[1:]):
            total_time += self.current_network[node_from][node_to]['current_travel_time']
            if total_time >= time_limit:
                return False
        return True

    def init_current_speeds(self):
        speeds = nx.get_edge_attributes(self.current_network, name='speed_kph')
        travel_times = {}
        for (node_from, node_to), speed in speeds.items():
            travel_times[(node_from, node_to)] = self.get_travel_time(node_from, node_to, speed)
        nx.set_edge_attributes(self.current_network, values=speeds, name="current_speed")
        nx.set_edge_attributes(self.current_network, values=travel_times, name="current_travel_time")

    def update_current_speeds(self, segments_to_update: Dict[SegmentId, Optional[SpeedKph]]):
        """
        This methods updates the current speeds and travel times for segments
        passed in `segments_to_update`.
        """
        max_speeds = nx.get_edge_attributes(self.current_network, name='speed_kph')

        new_current_speeds = {}
        for ((node_from, node_to), speed) in segments_to_update.items():
            edge = self.current_network[node_from][node_to]
            max_speed = max_speeds[(node_from, node_to)]
            speed = max_speed if speed is None else speed
            speed = min(speed, max_speed)
            speed = round_speed(speed)
            edge["current_speed"] = speed
            edge["current_travel_time"] = self.get_travel_time(node_from, node_to, speed)
            new_current_speeds[(node_from, node_to)] = speed

        return new_current_speeds

    def get_current_max_speed(self, node_from: int, node_to: int):
        return self.current_network[node_from][node_to]['speed_kph']

    def get_original_max_speed(self, node_from: int, node_to: int):
        return self.original_network[node_from][node_to]['speed_kph']

    def is_route_closed(self, route: List[int]):
        for node_from, node_to in zip(route[:-1], route[1:]):
            if self.get_current_max_speed(node_from, node_to) == 0:
                return True
        return False

    def set_data_dir(self, path):
        if self.data_dir is not None:
            cl.warn(f"The data dir has changed from '{self.data_dir}' to '{path}.")
        self.data_dir = path

    def get_osm_segment(self, node_from: int, node_to: int):
        data = self.original_network.get_edge_data(node_from, node_to)
        return Segment(
            node_from=node_from,
            node_to=node_to,
            length=data["length"],
            max_allowed_speed_kph=data["speed_kph"],
        )

    def osm_route_to_py_segments(self, osm_route: Route) -> List[Segment]:
        """Prepare list of segments based on route."""
        result = []
        for u, v in zip(osm_route[:-1], osm_route[1:]):
            result.append(self.get_osm_segment(u, v))
        return result

    def shortest_path_by_gps(self, gps_start, gps_end):
        """Compute shortest path between two gps points."""
        assert type(gps_start) == type(gps_end), \
            "Start and end gps has to be of the same type"

        if type(gps_start) == "list" or type(gps_start) == tuple:
            gps_start_ = list(gps_start)
            gps_end_ = list(gps_end)
        else:
            gps_start_ = [gps_start]
            gps_end_ = [gps_end]

        def nearest_nodes(gps):
            p = gps.point()
            return ox.distance.nearest_nodes(self.network, p.x, p.y)

        start_nodes = list(map(nearest_nodes, gps_start_))
        end_nodes = list(map(nearest_nodes, gps_end_))

        if len(start_nodes) == len(end_nodes) == 1:
            return self.shortest_path(start_nodes[0], end_nodes[0])
        return self.shortest_path(start_nodes, end_nodes)

    def shortest_path(self, origin, dest):
        """Compute shortest path between two OSM nodes."""
        try:
            return nx.shortest_path(self.original_network, origin, dest, weight="length")
        except NetworkXNoPath:
            return None

    def k_shortest_paths(self, origin, dest, k):
        """Compute k-shortest paths between two OSM nodes."""
        paths_gen = nx.shortest_simple_paths(G=self.current_network, source=origin,
                                             target=dest, weight=segment_weight)
        try:
            for path in itertools.islice(paths_gen, 0, k):
                yield path
        except NetworkXNoPath:
            return None

    def fastest_path(self, origin, dest):
        """Compute fastest path between two OSM nodes."""
        try:
            return nx.dijkstra_path(self.current_network, origin, dest, weight='current_travel_time')
        except NetworkXNoPath:
            return None

    def k_fastest_paths(self, origin, dest, k):
        """Compute k-fastest paths between two OSM nodes."""
        paths_gen = nx.shortest_simple_paths(G=self.current_network, source=origin,
                                             target=dest, weight="current_travel_time")
        try:
            for path in itertools.islice(paths_gen, 0, k):
                yield path
        except NetworkXNoPath:
            return None

    def osm_to_hdf5_id(self, id: int) -> int:
        return self.osm_to_hdf_map_ids[id]

    def hdf5_to_osm_id(self, id: int) -> int:
        return self.hdf_to_osm_map_ids[id]

    def _load(self):
        if self.file_path is None:
            cl.info("Map loaded from memory object.")
        elif os.path.exists(self.file_path):
            cl.info(f"Loading network for '{self.name}' from local map.")
            return load_graphml(self.file_path), False
        else:
            cl.info(f"Loading map for {self.name} via OSM API...")

            osmnx.settings.overpass_settings = f"[out:json][timeout:{{timeout}}][date:'{self.download_date}']"

            north, west, south, east = self.bbox.get_coords()
            network = graph_from_bbox(north, south, east, west,
                                      network_type="drive",
                                      retain_all=False)

            cl.info(f"{self.name}'s map loaded.")
            return network, True

    def _store(self):
        if self.file_path is None:
            cl.info("Map loaded from memory. "
                    "If you want to save it please set up valid data directory.")
            return

        if self.network is not None:
            save_graphml(self.network, self.file_path)
            cl.info(f"{self.name}'s map saved in {self.file_path}")

    def init_temporary_max_speeds(self, speeds_path):
        """Load temporary speeds from csv file."""
        timestamp_format = '%Y-%m-%d %H:%M:%S'

        with open(speeds_path) as f:
            reader = csv.reader(f, delimiter=';')
            next(reader, None)
            for row in reader:
                node_from, node_to, speed = int(row[0]), int(row[1]), float(row[2])
                speed = round_speed(speed)
                timestamp_from = datetime.strptime(row[3], timestamp_format)
                timestamp_to = datetime.strptime(row[4], timestamp_format)

                self.temporary_speeds.append(TemporarySpeed(node_from,
                                                            node_to,
                                                            SpeedKph(speed),
                                                            self.get_original_max_speed(node_from, node_to),
                                                            timestamp_from, timestamp_to, False))

        self.temporary_speeds.sort(key=lambda x: x.timestamp_from)

    def update_temporary_max_speeds(self, timestamp: datetime):
        """Update max speeds according to the temporary speeds."""
        new_max_speeds = {}
        check_segments = []
        ts_to_remove = []
        for ts in self.temporary_speeds:
            if timestamp > ts.timestamp_to:
                # the temporary speed is no longer valid, restore original speed
                new_max_speeds[(ts.node_from, ts.node_to)] = ts.original_max_speed
                ts_to_remove.append(ts)
                check_segments.append((ts.node_from, ts.node_to))
            elif ts.timestamp_from <= timestamp <= ts.timestamp_to and not ts.active:
                # the temporary speed is valid, apply it
                new_max_speeds[(ts.node_from, ts.node_to)] = ts.temporary_speed
                ts.active = True
                check_segments.append((ts.node_from, ts.node_to))
        for ts in ts_to_remove:
            self.temporary_speeds.remove(ts)

        nx.set_edge_attributes(self.current_network, values=new_max_speeds, name='speed_kph')

        new_current_speeds = {}
        new_travel_times = {}
        for (node_from, node_to) in check_segments:
            data = self.current_network.get_edge_data(node_from, node_to)
            max_speed = data['speed_kph']
            current_speed = data['current_speed']
            current_speed = current_speed if current_speed < max_speed else max_speed
            new_current_speeds[(node_from, node_to)] = current_speed
            new_travel_times[(node_from, node_to)] = self.get_travel_time(node_from, node_to, current_speed)

        nx.set_edge_attributes(self.current_network, values=new_current_speeds, name='current_speed')
        nx.set_edge_attributes(self.current_network, values=new_travel_times, name='current_travel_time')

        return new_current_speeds

    def has_temporary_speeds_planned(self):
        return len(self.temporary_speeds) > 0

    def get_hdf5_edge_id(self, segment_id: SegmentId) -> int:
        (node_from, node_to) = segment_id
        return get_edge_id_from_data(self.current_network[node_from][node_to])


def admin_level_to_road_filter(admin_level):  # TODO: where to put it?
    """Create a road filter based on administrative level."""
    if admin_level <= 2:  # country level
        return '["highway"~"motorway|trunk|primary"]'
    elif admin_level <= 6:  # county level
        return '["highway"~"motorway|trunk|primary|secondary"]'
    # elif admin_level <= 7:  # district level  # NOTE: the district level is
    # the lowest level currently,
    # then no restrictions
    #     return '["highway"~"motorway|trunk|primary|secondary|tertiary"]'
    else:
        return None  # all routes
