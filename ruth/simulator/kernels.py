import logging
import random
from typing import List, Optional, Tuple

from .ptdr import PTDRInfo
from ..data.map import Map, osm_routes_to_segment_ids
from ..data.segment import Route
from ..utils import is_root_debug_logging
from ..vehicle import Vehicle, VehicleAlternatives, VehicleRouteSelection
from ..zeromq.src.client import Message

AlternativeRoutes = List[Route]


# Alternatives
class AlternativesProvider:

    def __init__(self):
        self.vehicle_behaviour = None

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

    @staticmethod
    def remove_infinity_alternatives(alternatives: List[AlternativeRoutes],
                                     routing_map: Map) -> List[
        AlternativeRoutes]:
        """
        Removes alternatives that contain infinity.
        """
        filtered_alternatives = []
        for alternatives_for_vehicle in alternatives:
            for_vehicle = []
            for alternative in alternatives_for_vehicle:
                # calculate travel time for alternative
                if not routing_map.is_route_closed(alternative):
                    for_vehicle.append(alternative)
            filtered_alternatives.append(for_vehicle)
        return filtered_alternatives


class FastestPathsAlternatives(AlternativesProvider):

    def __init__(self):
        self.vehicle_behaviour = VehicleAlternatives.DIJKSTRA_FASTEST

    def compute_alternatives(self, map: Map, vehicles: List[Vehicle], k: int) -> List[
        Optional[AlternativeRoutes]]:
        return [vehicle.k_fastest_paths(k) for vehicle in vehicles]


class ShortestPathsAlternatives(AlternativesProvider):

    def __init__(self):
        self.vehicle_behaviour = VehicleAlternatives.DIJKSTRA_SHORTEST

    def compute_alternatives(self, map: Map, vehicles: List[Vehicle], k: int) -> List[
        Optional[AlternativeRoutes]]:
        return [vehicle.k_shortest_paths(k) for vehicle in vehicles]


class ZeroMQDistributedAlternatives(AlternativesProvider):
    from ..zeromq.src.client import Client

    def __init__(self, client: Client):
        self.vehicle_behaviour = VehicleAlternatives.PLATEAU_FASTEST
        self.client = client

    def load_map(self, map: Map):
        """
        Loads updated information from the passed map.
        """
        map_path = map.save_hdf()
        self.client.broadcast(Message(kind="load-map", data=map_path))

    def compute_alternatives(self, map: Map, vehicles: List[Vehicle], k: int) -> List[
        Optional[AlternativeRoutes]]:
        messages = [Message(kind="alternatives", data={
            "start": map.osm_to_hdf5_id(v.next_routing_od_nodes[0]),
            "destination": map.osm_to_hdf5_id(v.next_routing_od_nodes[1]),
            "max_routes": k
        }) for v in vehicles]

        # The formatting can be expensive, so we explicitly set if debug logging is enabled first
        if is_root_debug_logging():
            logging.debug(f"Sending alternatives to distributed worker: {messages}")

        results = self.client.compute(messages)

        if is_root_debug_logging():
            logging.debug(f"Response from worker: {results}")
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

    def __init__(self):
        self.vehicle_behaviour = None

    def select_routes(self, route_possibilities: List[VehicleWithPlans]) -> List[VehicleWithRoute]:
        raise NotImplementedError


class FirstRouteSelection(RouteSelectionProvider):
    """
    Selects the first route for each car.
    """
    def __init__(self):
        self.vehicle_behaviour = VehicleRouteSelection.FIRST

    def select_routes(self, route_possibilities: List[VehicleWithPlans]) -> List[VehicleWithRoute]:
        return [(vehicle, routes[0]) for (vehicle, routes) in route_possibilities]


class RandomRouteSelection(RouteSelectionProvider):

    def __init__(self, seed: int = None):
        self.vehicle_behaviour = VehicleRouteSelection.RANDOM
        self.generator = random.Random(seed)

    """
    Selects random route for each car.
    """

    def select_routes(self, route_possibilities: List[VehicleWithPlans]) -> List[VehicleWithRoute]:
        result = []
        for (vehicle, routes) in route_possibilities:
            route = self.generator.choice(routes)
            result.append((vehicle, route))
        return result


class ZeroMQDistributedPTDRRouteSelection(RouteSelectionProvider):
    from ..zeromq.src.client import Client

    def __init__(self, client: Client):
        self.vehicle_behaviour = VehicleRouteSelection.PTDR
        self.client = client
        self.ptdr_info = None

    def update_segment_profiles(self, ptdr_info: PTDRInfo):
        self.ptdr_info = ptdr_info
        # self.client.broadcast(Message(kind="load-profiles", data=str(path)))

    """
    Sends routes to a distributed node that calculates a Monte Carlo simulation and returns
    the shortest route for each car.
    """

    def select_routes(self, route_possibilities: List[VehicleWithPlans]) -> List[VehicleWithRoute]:
        messages = [Message(kind="monte-carlo", data={
            "routes": osm_routes_to_segment_ids(routes),
            "frequency": vehicle.frequency.total_seconds(),
            "departure_time": self.ptdr_info.get_time_from_start_of_interval(vehicle.time_offset),
        }) for (vehicle, routes) in route_possibilities]

        if is_root_debug_logging():
            logging.debug(f"Sending PTDR to distributed worker: {messages}")

        shortest_routes = self.client.compute(messages)

        if is_root_debug_logging():
            logging.debug(f"Response from worker: {shortest_routes}")

        return [(vehicle, routes[shortest_route]) for ((vehicle, routes), shortest_route) in
                zip(route_possibilities, shortest_routes)]
