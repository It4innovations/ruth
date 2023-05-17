import json
import logging

import zmq
import networkx as nx
import itertools
import osmnx as ox

from osmnx import graph_from_polygon, load_graphml, save_graphml
from networkx.exception import NetworkXNoPath


logger = logging.getLogger(__name__)


def segment_weight(n1, n2, data):
    assert "length" in data, f"Expected the 'length' of segment to be known. ({n1}, {n2})"
    return float(data['length']) + float(f"0.{n1}{n2}")


def load(fname):
    network = load_graphml(fname)
    return ox.get_digraph(network)


class Worker:
    def __init__(self, map, address="localhost", port: int = 5560):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.DEALER)
        self.address = f"tcp://{address}:{port}"
        self.socket.connect(self.address)
        self.map = load(map)
        logger.info("Worker is initialized")

    def get_k_paths(self, message):
        origin, dest, k = message[0], message[1], message[2]
        logger.info(f"Executing alternatives query {origin}->{dest} with k={k}")
        paths_gen = nx.shortest_simple_paths(G=self.map, source=origin, target=dest,
                                             weight=segment_weight)
        try:
            for path in itertools.islice(paths_gen, 0, k):
                yield path
        except NetworkXNoPath:
            return None

    def run(self):
        while True:
            id, message = self.socket.recv_multipart()

            # Decode
            message = json.loads(message.decode())

            # Compute
            result = list(self.get_k_paths(message))

            # Serialize
            result = json.dumps(result).encode()

            # Send
            self.socket.send_multipart([
                id,
                result
            ])
