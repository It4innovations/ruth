import json
from typing import List, Optional

from ..vehicle import Vehicle

Route = List[int]
AlternativeRoutes = List[Route]


# Alternatives
class AlternativesProvider:
    """
    Provides implementation of computing alternatives.
    For a given list of vehicles and parameter `k`, computes a list of routes for each vehicle.
    """
    def compute_alternatives(self, vehicles: List[Vehicle], k: int) -> List[Optional[AlternativeRoutes]]:
        raise NotImplementedError


class FastestPathsAlternatives(AlternativesProvider):
    def compute_alternatives(self, vehicles: List[Vehicle], k: int) -> List[Optional[AlternativeRoutes]]:
        return [vehicle.k_fastest_paths(k) for vehicle in vehicles]


class ShortestPathsAlternatives(AlternativesProvider):
    def compute_alternatives(self, vehicles: List[Vehicle], k: int) -> List[Optional[AlternativeRoutes]]:
        return [vehicle.k_shortest_paths(k) for vehicle in vehicles]


class ZeroMQDistributedAlternatives(AlternativesProvider):
    def __init__(self, port: int):
        from ..zeromq.src.client import Client
        self.client = Client(port=port)

    def compute_alternatives(self, vehicles: List[Vehicle], k: int) -> List[Optional[AlternativeRoutes]]:
        od_paths = [(*v.next_routing_od_nodes, k) for v in vehicles]
        inputs = [json.dumps(x).encode() for x in od_paths]
        result = self.client.compute(inputs)
        return result
