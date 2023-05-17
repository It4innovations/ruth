import json
from typing import List

from ..vehicle import Vehicle


class KernelProvider:
    """
    Provides implementation of computationally intensive tasks.
    """
    def compute_alternatives(self, vehicles: List[Vehicle], k: int) -> List[List[List[int]]]:
        raise NotImplementedError


class LocalKernelProvider(KernelProvider):
    def compute_alternatives(self, vehicles: List[Vehicle], k: int) -> List[List[List[int]]]:
        result = []
        for vehicle in vehicles:
            result.append(vehicle.k_shortest_paths(k))
        return result


class ZeroMqKernelProvider(KernelProvider):
    def __init__(self, port: int):
        from ..zeromq.src.client import Client
        self.client = Client(port=port)

    def compute_alternatives(self, vehicles: List[Vehicle], k: int) -> List[List[List[int]]]:
        od_paths = [(*v.next_routing_od_nodes, k) for v in vehicles]
        inputs = [json.dumps(x).encode() for x in od_paths]
        result = self.client.compute(inputs)
        return result
