from typing import List, Optional, Tuple
import random

from ..data.segment import Route
from ..data.map import Map
from ..vehicle import Vehicle
from ..zeromq.src.client import Message

AlternativeRoutes = List[Route]


# Alternatives
class AlternativesProvider:
    def load_map(self, map: Map):
        """
        Loads updated information from the passed map.
        """
        pass

    def compute_alternatives(self, map: Map, vehicles: List[Vehicle], k: int) -> List[
        Optional[AlternativeRoutes]]:
        """
        Provides implementation of computing alternatives.
        For a given list of vehicles and parameter `k`, computes a list of routes for each vehicle.
        """
        raise NotImplementedError


class FastestPathsAlternatives(AlternativesProvider):
    def compute_alternatives(self, map: Map, vehicles: List[Vehicle], k: int) -> List[
        Optional[AlternativeRoutes]]:
        return [vehicle.k_fastest_paths(k) for vehicle in vehicles]


class ShortestPathsAlternatives(AlternativesProvider):
    def compute_alternatives(self, map: Map, vehicles: List[Vehicle], k: int) -> List[
        Optional[AlternativeRoutes]]:
        return [vehicle.k_shortest_paths(k) for vehicle in vehicles]


class ZeroMQDistributedAlternatives(AlternativesProvider):
    def __init__(self, port: int):
        from ..zeromq.src.client import Client
        self.client = Client(port=port)

    def load_map(self, map: Map):
        """
        Loads updated information from the passed map.
        """
        # TODO: implement with broadcast
        map_path = map.save_hdf()
        self.client.compute([Message(kind="load-map", data=map_path)])

    def compute_alternatives(self, map: Map, vehicles: List[Vehicle], k: int) -> List[
        Optional[AlternativeRoutes]]:
        messages = [Message(kind="alternatives", data={
            "start": map.osm_to_hdf5_id(v.next_routing_od_nodes[0]),
            "destination": map.osm_to_hdf5_id(v.next_routing_od_nodes[1]),
            "max_routes": k
        }) for v in vehicles]
        # logging.info(f"Sending alternatives to distributed worker: {messages}")
        results = self.client.compute(messages)
        # logging.info(f"Response from worker: {results}")
        remapped_routes = [
            [[map.hdf5_to_osm_id(node_id) for node_id in route] for route in result["routes"]]
            for result in results
        ]
        return remapped_routes


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
        return [(vehicle, routes[0]) for (vehicle, routes) in route_possibilities]


class RandomRouteSelection(RouteSelectionProvider):

    def __init__(self):
        # TODO: add seed from config
        self.generator = random.Random(1)

    """
    Selects random route for each car.
    """

    def select_routes(self, route_possibilities: List[VehicleWithPlans]) -> List[VehicleWithRoute]:
        result = []
        for (vehicle, routes) in route_possibilities:
            route = self.generator.choice(routes)
            result.append((vehicle, route))
        return result
