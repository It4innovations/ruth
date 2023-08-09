"""Routing map module."""
import itertools
import os
import pickle

import networkx as nx
import osmnx as ox
from osmnx import graph_from_place, load_graphml, save_graphml
from networkx.exception import NetworkXNoPath

from .hdf5_writer import save_graph_to_hdf5
from ..log import console_logger as cl
from ..metaclasses import Singleton


def segment_weight(n1, n2, data):
    return float(data['length']) + float(f"0.{n1}{n2}")


def segment_weight_speed(n1, n2, data):
    return float(data['speed_kph'])


def save(G, fname):
    nx.write_gml(G, fname)


class Map(metaclass=Singleton):
    """Routing map."""

    def __init__(self, border, data_dir="./data", with_speeds=True):
        """Initialize a map via the border.

        If `data_dir` provided, the map is loaded from locally stored maps preferably.
        """
        self.border = border
        self.data_dir = data_dir
        self.network, fresh_data = self._load()
        self.simple_network = ox.get_digraph(self.network)
        if with_speeds:
            self.network = ox.add_edge_speeds(self.network)
        if fresh_data:
            self._store()
        self.ids_from_64_to_32 = self.to_hdf5()

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

    def set_data_dir(self, path):
        if self.data_dir is not None:
            cl.warn(f"The data dir has changed from '{self.data_dir}' to '{path}.")
        self.data_dir = path

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

    def to_hdf5(self):
        return save_graph_to_hdf5(self.simple_network, 'data/map.hdf5')

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
