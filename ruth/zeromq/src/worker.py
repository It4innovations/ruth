import json
import zmq
import networkx as nx
import itertools

from networkx.exception import NetworkXNoPath


def segment_weight(n1, n2, data):
    assert "length" in data, f"Expected the 'length' of segment to be known. ({n1}, {n2})"
    return float(data['length']) + float(f"0.{n1}{n2}")


class Worker:
    def __init__(self, map, address="localhost", port: int = 5560):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.DEALER)
        self.address = f"tcp://{address}:{port}"
        self.socket.connect(self.address)
        self.map = map

    def get_k_paths(self, message):
        origin, dest, k = message[0], message[1], message[2]
        paths_gen = nx.shortest_simple_paths(G=self.map, source=origin, target=dest, weight=segment_weight)
        try:
            for path in itertools.islice(paths_gen, 0, k):
                yield path
        except NetworkXNoPath:
            return None

    def run(self):
        message_id = 0
        while True:
            id, message = self.socket.recv_multipart()

            # Decode
            message = json.loads(message.decode())

            # Compute
            result = self.get_k_paths(message)

            # Serialize
            result = json.dumps(result).encode()

            # Send
            self.socket.send_multipart([
                id,
                result
            ])

            message_id += 1
            #print(f'Sending {message_id}')
