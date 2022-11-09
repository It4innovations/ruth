"""Routing map module."""
import itertools
import os
from math import floor

import networkx as nx
import osmnx as ox
from osmnx import graph_from_polygon, load_graphml, save_graphml
from networkx.exception import NetworkXNoPath

from ..log import console_logger as cl
from ..metaclasses import Singleton


def segment_weight(n1, n2, data):
    assert "length" in data, f"Expected the 'length' of segment to be known. ({n1}, {n2})"
    return float(data['length']) + float(f"0.{n1}{n2}")


def path_with_similar_prefix(p1, p2, prefix_size_ratio):
    """Compares two paths on a prefix specified as a ratio between 0.0 and 1.0. If there is a difference on the
    nodes at the same position the paths are considered not similar otherwise they are similar. The aim is to
    find alternatives that differ at the beginning."""

    assert 0.0 <= prefix_size_ratio <= 1.0, "Prefix size ration out of range."
    prefix_size = floor(len(p1) * prefix_size_ratio)
    prefix_size_ = prefix_size if prefix_size < len(p2) else len(p2)

    for i in range(prefix_size_):
        if p1[i] != p2[i]:
            return False
    return True


class Map(metaclass=Singleton):
    """Routing map."""

    def __init__(self, border, data_dir="./data", with_speeds=False):
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

    @property
    def file_path(self):
        """Path to locally stored map."""
        return os.path.join(self.data_dir, f"{self.name}.graphml")

    @property
    def name(self):
        """Name of the map."""
        return self.border.name

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

    def k_shortest_paths(self, origin, dest, k, default_path=None):
        """Compute k-shortest paths between two OSM nodes."""
        paths_gen = nx.shortest_simple_paths(G=self.simple_network, source=origin,
                                             target=dest, weight=segment_weight)

        try:
            alternatives = [] if default_path is None else [default_path[:]]
            i = 0
            for path in paths_gen:
                # TODO: decrease the ratio from 1.0 to something closer 0.2 to compare only a prefix
                if all(not path_with_similar_prefix(path, other, 1.0) for other in alternatives):
                    alternatives.append(path)
                    i += 1
                if i >= k:
                    break
            return alternatives
        except NetworkXNoPath:
            return None

    def _load(self):
        print(f"Map file path: {self.file_path}")
        if os.path.exists(self.file_path):
            cl.info(f"Loading network for '{self.name}' from local map.")
            return (load_graphml(self.file_path), False)
        else:
            cl.info(f"Loading map for {self.name} via OSM API...")
            network = graph_from_polygon(
                self.border.polygon(),
                network_type="drive",  # TODO: into config
                retain_all=True,
                clean_periphery=False,
                custom_filter=admin_level_to_road_filter(self.border.admin_level),
            )
            cl.info(f"{self.name}'s map loaded.")
            return (network, True)

    def _store(self):
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
