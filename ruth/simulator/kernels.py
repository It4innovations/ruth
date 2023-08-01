import json
from typing import List, Optional, Tuple

from .segment import Route
from ..vehicle import Vehicle

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
        inputs = [json.dumps({
            "start": v.next_routing_od_nodes[0],
            "destination": v.next_routing_od_nodes[1],
            "max_routes": k
        }).encode() for v in vehicles]
        result = self.client.compute(kernel="alternatives", inputs=inputs)
        return result


VehicleWithPlans = Tuple[Vehicle, AlternativeRoutes]
VehicleWithRoute = Tuple[Vehicle, Route]

# Route selection
class RouteSelectionProvider:
    """
    Provides implementation of route selection.
    For a given list of alternatives (k per vehicle), select the best route for each vehicle.
    """
    def select_routes(self, route_possibilities: List[VehicleWithPlans]) -> List[VehicleWithRoute]:
        raise NotImplementedError


class FirstRouteSelection(RouteSelectionProvider):
    """
    Selects the first route for each car.
    """
    def select_routes(self, route_possibilities: List[VehicleWithPlans]) -> List[VehicleWithRoute]:
        # TODO: react better to situations where no route is found
        return [(vehicle, routes[0]) for (vehicle, routes) in route_possibilities]
