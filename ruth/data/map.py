"""Routing map module."""
import csv
import itertools
import os
import pickle
from pathlib import Path
from typing import List

import networkx as nx
import osmnx as ox
from osmnx import graph_from_place, load_graphml, save_graphml
from networkx.exception import NetworkXNoPath

from .hdf5_writer import save_graph_to_hdf5
from ..log import console_logger as cl
from ..metaclasses import Singleton
from ..simulator.segment import Route, Segment


def segment_weight(n1, n2, data):
    return float(data['length']) + float(f"0.{n1}{n2}")


def segment_weight_speed(n1, n2, data):
    return float(data['current_travel_time'])


def save(G, fname):
    nx.write_gml(G, fname)


def get_osm_segment_id(node_from: int, node_to: int):
    return f"OSM{node_from}T{node_to}"


class Map(metaclass=Singleton):
    """Routing map."""

    def __init__(self, border, data_dir="./data", with_speeds=True):
        """Initialize a map via the border.

        If `data_dir` provided, the map is loaded from locally stored maps preferably.
        """
        self.border = border
        self.data_dir = data_dir
        self.network, fresh_data = self._load()
        if with_speeds:
            self.network = ox.add_edge_speeds(self.network)
            self.init_current_speeds()

        self.simple_network = ox.get_digraph(self.network)

        if fresh_data:
            self._store()
        self.hdf5_file_path = str((Path(data_dir) / "map.hdf5").absolute())
        self.osm_to_hdf_map_ids = self.to_hdf5(self.hdf5_file_path)
        self.hdf_to_osm_map_ids = {v: k for (k, v) in self.osm_to_hdf_map_ids.items()}
        assert len(self.osm_to_hdf_map_ids) == len(self.hdf_to_osm_map_ids)

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
        return self.border.name

    def init_current_speeds(self):
        speeds = nx.get_edge_attributes(self.network, name='speed_kph')
        lengths = nx.get_edge_attributes(self.network, name='length')
        travel_times = {}
        for key, value in speeds.items():
            travel_times[key] = float(lengths[key]) * 3.6 / float(value)
        nx.set_edge_attributes(self.network, values=speeds, name="current_speed")
        nx.set_edge_attributes(self.network, values=travel_times, name="current_travel_time")

    def update_current_speeds(self, segment_ids, speeds):
        new_speeds = {}
        new_travel_times = {}
        lengths = nx.get_edge_attributes(self.simple_network, name='length')
        for (node_from, node_to), speed in zip(segment_ids, speeds):
            new_speeds[(node_from, node_to)] = speed
            travel_time = float('inf') if speed == 0 else float(lengths[(node_from, node_to)]) * 3.6 / float(speed)
            new_travel_times[(node_from, node_to)] = travel_time
        nx.set_edge_attributes(self.simple_network, values=new_speeds, name='current_speed')
        nx.set_edge_attributes(self.simple_network, values=new_travel_times, name='current_travel_time')

    def get_segment_max_speed(self, node_from, node_to):
        return self.simple_network[node_from][node_to]['speed_kph']

    def set_data_dir(self, path):
        if self.data_dir is not None:
            cl.warn(f"The data dir has changed from '{self.data_dir}' to '{path}.")
        self.data_dir = path

    def osm_route_to_py_segments(self, osm_route: Route) -> List[Segment]:
        """Prepare list of segments based on route."""
        edge_data = ox.utils_graph.get_route_edge_attributes(self.network,
                                                             osm_route)
        edges = zip(osm_route, osm_route[1:])
        return [
            Segment(
                id=get_osm_segment_id(from_, to_),
                length=data["length"],
                max_allowed_speed_kph=data["speed_kph"],
            )
            # NOTE: the zip is correct as the starting node_id is of the interest
            for i, ((from_, to_), data) in enumerate(zip(edges, edge_data))
        ]

    def shortest_path_by_gps(self, gps_start, gps_end):
        """Compute shortest path between two gps points."""
        assert type(gps_start) == type(gps_end), \
            "Start and end gps has to be of the same type"

        if type(gps_start) == "list" or type(gps_start) == tuple:
            gps_start_ = list(gps_start)
            gps_end = list(gps_end)
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
        return ox.shortest_path(self.network, origin, dest)

    def k_shortest_paths(self, origin, dest, k):
        """Compute k-shortest paths between two OSM nodes."""
        paths_gen = nx.shortest_simple_paths(G=self.simple_network, source=origin,
                                             target=dest, weight=segment_weight)
        try:
            for path in itertools.islice(paths_gen, 0, k):
                yield path
        except NetworkXNoPath:
            return None

    def fastest_path(self, origin, dest):
        """Compute fastest path between two OSM nodes."""
        return nx.dijkstra_path(self.network, origin, dest, weight=segment_weight_speed)

    def k_fastest_paths(self, origin, dest, k):
        """Compute k-fastest paths between two OSM nodes."""
        paths_gen = nx.shortest_simple_paths(G=self.simple_network, source=origin,
                                             target=dest, weight=segment_weight_speed)
        try:
            for path in itertools.islice(paths_gen, 0, k):
                yield path
        except NetworkXNoPath:
            return None

    def to_hdf5(self, path: str):
        return save_graph_to_hdf5(self.simple_network, path)

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
            # # Change from "graph_from_polygon"
            # network = graph_from_polygon(
            #     self.border.polygon(),
            #     network_type="drive",  # TODO: into config
            #     retain_all=True,
            #     clean_periphery=False,
            #     custom_filter=admin_level_to_road_filter(self.border.admin_level),
            # )

            # Change to "graph_from_place"
            network = graph_from_place(
                'Prague, Czech republic',
                network_type="drive",  # TODO: into config
                retain_all=True,
                clean_periphery=False,
                custom_filter=admin_level_to_road_filter(self.border.admin_level),
            )
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

    def update_speeds_from_file(self, speeds_path):
        """Update max speed on segment based on file config."""
        new_speeds = {}
        new_travel_times = {}
        lengths = nx.get_edge_attributes(self.simple_network, name='length')

        with open(speeds_path, newline='') as f:
            reader = csv.reader(f, delimiter=';')
            next(reader, None)
            for row in reader:
                node_from, node_to, speed = int(row[0]), int(row[1]), float(row[2])

            new_speeds[(node_from, node_to)] = speed
            travel_time = float('inf') if speed == 0 else float(lengths[(node_from, node_to)]) * 3.6 / float(speed)
            new_travel_times[(node_from, node_to)] = travel_time

        nx.set_edge_attributes(self.simple_network, values=new_speeds, name='speed_kph')
        nx.set_edge_attributes(self.simple_network, values=new_speeds, name='current_speed')
        nx.set_edge_attributes(self.simple_network, values=new_travel_times, name='current_travel_time')


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