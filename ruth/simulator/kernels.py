import logging
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from .ptdr import PTDRInfo
from ..data.map import Map, osm_routes_to_segment_ids
from ..data.segment import Route, SegmentId, SpeedMps, RouteWithTime
from ..utils import is_root_debug_logging
from ..vehicle import Vehicle, VehicleAlternatives, VehicleRouteSelection

AlternativeRoutes = List[RouteWithTime]

VehicleWithPlans = Tuple[Vehicle, AlternativeRoutes]
VehicleWithRoute = Tuple[Vehicle, RouteWithTime]


# Alternatives
class AlternativesProvider:

    def __init__(self):
        self.vehicle_behaviour = None
        self.routing_map = None

    def load_map(self, routing_map: Map):
        """
        Loads updated information from the passed map.
        """
        self.routing_map = routing_map

    def update_map(self, segments: Dict[SegmentId, Optional[SpeedMps]]):
        """
        Update speeds of the passed segments in a previously passed map.
        """
        pass

    def compute_alternatives(self, vehicles: List[Vehicle], k: int) -> List[
        Optional[AlternativeRoutes]]:
        """
        Provides implementation of computing alternatives.
        For a given list of vehicles and parameter `k`, computes a list of routes for each vehicle.
        """
        raise NotImplementedError

    def get_routes_travel_times(self, routes: List[Route]) -> List[Optional[float]]:
        """
        Returns the travel times for the given routes.
        """
        raise NotImplementedError


class FastestPathsAlternatives(AlternativesProvider):

    def __init__(self):
        self.vehicle_behaviour = VehicleAlternatives.DIJKSTRA_FASTEST

    def compute_alternatives(self, vehicles: List[Vehicle], k: int) -> List[
        Optional[AlternativeRoutes]]:
        return [[(route, None) for route in vehicle.k_fastest_paths(k, self.routing_map)]
                for vehicle in vehicles]

    def get_routes_travel_times(self, routes: List[Route]) -> List[Optional[float]]:
        travel_times = [self.routing_map.get_path_travel_time(route) for route in routes]
        return travel_times


class ShortestPathsAlternatives(AlternativesProvider):

    def __init__(self):
        self.vehicle_behaviour = VehicleAlternatives.DIJKSTRA_SHORTEST

    def compute_alternatives(self, vehicles: List[Vehicle], k: int) -> List[
        Optional[AlternativeRoutes]]:
        return [[(route, None) for route in vehicle.k_shortest_paths(k, self.routing_map)]
                for vehicle in vehicles]

    def get_routes_travel_times(self, routes: List[Route]) -> List[Optional[float]]:
        return len(routes) * [None]


class MPIDistributedAlternatives(AlternativesProvider):
    def __init__(self):
        super().__init__()
        self.vehicle_behaviour = VehicleAlternatives.PLATEAU_FASTEST
        self.routing_map = None
        self.map_id = None

        try:
            import ruthlib as ru
            self.ru = ru
        except ImportError:
            raise ImportError("ruthlib is not installed. Please install ruthlib to use Plateau Alternatives.")

    def load_map(self, routing_map: Map):
        """
        Loads updated information from the passed map.
        """
        map_path = routing_map.save_hdf()
        self.ru.setup_map(map_path)
        self.routing_map = routing_map
        self.map_id = 0

    def update_map(self, segments: Dict[SegmentId, Optional[SpeedMps]]):
        """
        Update speeds of the passed segments in a previously passed map.
        """
        message = []
        for segment_id, speed in segments.items():
            edge_id = self.routing_map.get_hdf5_edge_id(segment_id)
            speed = speed if speed is not None else self.routing_map.get_current_max_speed(segment_id[0], segment_id[1])
            message.append((edge_id, speed))

        self.ru.update_speeds(message)
        return

    def postprocess(self, li, vehicles):
        id_to_index = {vid: i for i, vid in enumerate(li[0])}
        remapped_routes = []
        for vehicle_index in range(len(vehicles)):
            i = id_to_index.get(vehicle_index)
            if i is not None:
                results = [
                    ([self.routing_map.hdf5_to_osm_id(node_id) for node_id in route],travel_time)
                    for route, travel_time in zip(li[1][i], li[2][i])
                ]
                remapped_routes.append(results)
            else:
                remapped_routes.append([])
        return remapped_routes

    def compute_alternatives(self, vehicles: List[Vehicle], k: int) -> List[
        Optional[AlternativeRoutes]]:
        OD_matrix = [
            (self.routing_map.osm_to_hdf5_id(v.next_routing_od_nodes[0]),
                self.routing_map.osm_to_hdf5_id(v.next_routing_od_nodes[1]))
            for v in vehicles
        ]
        self.ru.do_alternatives(OD_matrix, k)

        remapped_routes = []

        li = self.ru.get_routes()

        # li[0] is a list of vehicle IDs, li[1] is a list of routes, and li[2] is a list of travel times
        remapped_routes = self.postprocess(li, vehicles)

        return remapped_routes

    def get_routes_travel_times(self, routes: List[Route]) -> List[Optional[float]]:
        routes = [
            [self.routing_map.osm_to_hdf5_id(node_id) for node_id in route] for route in routes
        ]
        self.ru.do_travel_times(routes)

        travel_times = self.ru.get_travel_times()
        travel_times = [tt if tt >= 0 else np.inf for tt in travel_times]

        return travel_times

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
    # from ..zeromq.src.client import Client

    def __init__(self, client):
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
        raise NotImplementedError("PTDR route selection is not implemented yet.")
        # messages = [Message(kind="monte-carlo", data={
        #     "routes": osm_routes_to_segment_ids(routes[0]),
        #     "frequency": vehicle.frequency.total_seconds(),
        #     "departure_time": self.ptdr_info.get_time_from_start_of_interval(vehicle.time_offset),
        # }) for (vehicle, routes) in route_possibilities]
        #
        # if is_root_debug_logging():
        #     logging.debug(f"Sending PTDR to distributed worker: {messages}")
        #
        # shortest_routes = self.client.compute(messages)
        #
        # if is_root_debug_logging():
        #     logging.debug(f"Response from worker: {shortest_routes}")
        #
        # return [(vehicle, routes[shortest_route]) for ((vehicle, routes), shortest_route) in
        #         zip(route_possibilities, shortest_routes)]
