import logging
import random
from typing import List, Optional, Tuple

from .ptdr import SegmentPTDRData
from ..data.map import Map
from ..data.segment import Route
from ..utils import is_root_debug_logging
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
    from ..zeromq.src.client import Client

    def __init__(self, client: Client):
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

    # FIXME: it's a hack to have this method here, but for now it's OK
    def update_segment_profiles(self, segments: List[SegmentPTDRData]):
        pass

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


class ZeroMQDistributedPTDRRouteSelection(RouteSelectionProvider):
    from ..zeromq.src.client import Client

    def __init__(self, client: Client):
        self.client = client

    def update_segment_profiles(self, segments: List[SegmentPTDRData]):
        self.client.broadcast(Message(kind="load-profiles", data=[{
            "id": segment.id,
            "length": segment.length,
            "speed": segment.max_speed,
            "profiles": [{
                "values": profile.values,
                "cumprobs": profile.cumprobs
            } for profile in segment.profiles],
        } for segment in segments]))

    """
    Sends routes to a distributed node that calculates a Monte Carlo simulation and returns
    the shortest route for each car.
    """
    def select_routes(self, route_possibilities: List[VehicleWithPlans]) -> List[VehicleWithRoute]:
        # TODO: calculate the correct time offset for Monte Carlo
        messages = [Message(kind="monte-carlo", data={
            "routes": routes,
            "frequency": vehicle.frequency,
            "departure_time": vehicle.time_offset
        }) for (vehicle, routes) in route_possibilities]

        if is_root_debug_logging():
            logging.debug(f"Sending PTDR to distributed worker: {messages}")

        shortest_routes = self.client.compute(messages)

        if is_root_debug_logging():
            logging.debug(f"Response from worker: {shortest_routes}")

        return [routes[shortest_routes] for ((vehicle, routes), shortest_route) in
                zip(route_possibilities, shortest_routes)]
