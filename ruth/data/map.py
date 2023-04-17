"""Routing map module."""
import itertools
import os
import pickle

import networkx as nx
import osmnx as ox
import json
from osmnx import graph_from_polygon, load_graphml, save_graphml
from networkx.exception import NetworkXNoPath
from cluster.cluster import Cluster, start_process

from ..log import console_logger as cl
from ..metaclasses import Singleton
from ..zeromq.src import client, worker


def segment_weight(n1, n2, data):
    assert "length" in data, f"Expected the 'length' of segment to be known. ({n1}, {n2})"
    return float(data['length']) + float(f"0.{n1}{n2}")


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

        # ------------------ ZeroMQ Init ------------------
        # Worker
        address = 'localhost'
        port = 5559
        WORK_DIR = '/home/user3574/PycharmProjects/ruth/ruth/zeromq'

        self.client = client.Client(port=port)
        for i in range(8):
            start_process(
                commands=[f"python3 {WORK_DIR}/ex_worker.py --address {'tcp://localhost:5559'} --port {5560} --map {map}"],
                workdir=str('tmp/workers'),
                pyenv=str('/home/user3574/PycharmProjects/ruth/venv'),
                modules=None,
                hostname=address,
                name=f"worker_{i}"
            )

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

    def k_shortest_paths_mq(self, origins, dests, k):
        # Encode into array
        array = [[origins[i], dests[i], k] for i in range(len(origins))]
        array = [json.dumps(x).encode() for x in array]

        return self.client.compute(array)

    def _load(self):
        if self.file_path is None:
            cl.info("Map loaded from memory object.")
        elif os.path.exists(self.file_path):
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
