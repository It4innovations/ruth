import logging
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from .ptdr import PTDRInfo
from ..data.map import Map
from ..data.segment import Route, SegmentId, SpeedMps, RouteWithTime
from ..vehicle import Vehicle, VehicleAlternatives, VehicleRouteSelection

AlternativeRoutes = List[RouteWithTime]

VehicleWithPlans = Tuple[Vehicle, AlternativeRoutes]
VehicleWithRoute = Tuple[Vehicle, RouteWithTime]

logger = logging.getLogger(__name__)


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

    def compute_alternatives(self, vehicles: List[Vehicle], k: int, use_origin_speeds: bool = False) -> List[
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
        super().__init__()
        self.vehicle_behaviour = VehicleAlternatives.DIJKSTRA_FASTEST

    def compute_alternatives(self, vehicles: List[Vehicle], k: int, use_origin_speeds: bool = False) -> List[
        Optional[AlternativeRoutes]]:
        return [[(route, None) for route in vehicle.k_fastest_paths(k, self.routing_map)]
                for vehicle in vehicles]

    def get_routes_travel_times(self, routes: List[Route]) -> List[Optional[float]]:
        travel_times = [self.routing_map.get_path_travel_time(route) for route in routes]
        return travel_times


class ShortestPathsAlternatives(AlternativesProvider):

    def __init__(self):
        super().__init__()
        self.vehicle_behaviour = VehicleAlternatives.DIJKSTRA_SHORTEST

    def compute_alternatives(self, vehicles: List[Vehicle], k: int, use_origin_speeds: bool = False) -> List[
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
        self._edge_id_cache = {}
        self._osm_lut = None

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
        # Per-map caches: segment -> HDF5 edge id, and the HDF5 -> OSM node
        # lookup table for vectorized route remapping.
        self._edge_id_cache = {}
        self._osm_lut = None

    def _osm_id_lut(self):
        if self._osm_lut is None:
            mapping = self.routing_map.hdf_to_osm_map_ids
            lut = np.empty(max(mapping.keys()) + 1, dtype=np.int64)
            for hdf_id, osm_id in mapping.items():
                lut[hdf_id] = osm_id
            self._osm_lut = lut
        return self._osm_lut

    def update_map(self, segments: Dict[SegmentId, Optional[SpeedMps]]):
        """
        Update speeds of the passed segments in a previously passed map.
        """
        n = len(segments)
        ids_array = np.empty(n, dtype=np.int32)
        speeds_array = np.empty(n, dtype=np.float32)
        edge_id_cache = self._edge_id_cache
        get_edge_id = self.routing_map.get_hdf5_edge_id
        get_speed = self.routing_map.get_current_max_speed

        for i, (segment_id, speed) in enumerate(segments.items()):
            edge_id = edge_id_cache.get(segment_id)
            if edge_id is None:
                edge_id = edge_id_cache[segment_id] = get_edge_id(segment_id)
            ids_array[i] = edge_id
            speeds_array[i] = speed if speed is not None else get_speed(segment_id[0], segment_id[1])

        self.ru.update_speeds(ids_array, speeds_array)
        return

    def postprocess(self, li, vehicles):
        """
        Remap HDF5 node IDs to OSM IDs.
        C++ returns routes sorted by vehicle ID (0, 1, 2, ...), so indices match directly.
        """
        vehicle_ids, routes_per_vehicle, travel_times_per_vehicle = li
        num_returned = len(vehicle_ids)

        # Defensive check: ensure C++ returned the expected number of results
        if num_returned != len(vehicles):
            raise ValueError(f"C++ returned {num_returned} results but expected {len(vehicles)} vehicles")

        remapped_routes = []
        for vehicle_index in range(len(vehicles)):
            # Verify vehicle ID matches expected index (defensive programming)
            if vehicle_ids[vehicle_index] != vehicle_index:
                raise ValueError(
                    f"Vehicle ID mismatch at index {vehicle_index}: "
                    f"expected {vehicle_index}, got {vehicle_ids[vehicle_index]}"
                )

            # Remap HDF5 node IDs to OSM IDs for each route (empty list if no routes found)
            results = [
                ([self.routing_map.hdf5_to_osm_id(node_id) for node_id in route], travel_time)
                for route, travel_time in zip(routes_per_vehicle[vehicle_index], travel_times_per_vehicle[vehicle_index])
            ]
            remapped_routes.append(results)

        return remapped_routes

    def postprocess_flat(self, flat_routes, vehicles):
        v_ids = flat_routes['vehicle_ids']

        if not np.all(v_ids[:-1] <= v_ids[1:]):
            raise ValueError("vehicle_ids are not sorted as expected.")

        id_to_idx = {vid: i for i, vid in enumerate(v_ids)}

        remapped_routes = []

        r_offsets = flat_routes['route_offsets']
        n_offsets = flat_routes['node_offsets']
        nodes = flat_routes['nodes']
        t_times = flat_routes['travel_times']

        # Remap every node HDF5 -> OSM in one vectorized gather instead of a
        # Python dict lookup per node.
        osm_nodes_all = self._osm_id_lut()[nodes]

        for v_idx in range(len(vehicles)):
            flat_v_idx = id_to_idx.get(v_idx)

            if flat_v_idx is not None:
                vehicle_results = []
                r_start, r_end = r_offsets[flat_v_idx], r_offsets[flat_v_idx + 1]

                for r in range(r_start, r_end):
                    n_start, n_end = n_offsets[r], n_offsets[r + 1]

                    travel_time = t_times[r]
                    osm_nodes = osm_nodes_all[n_start:n_end].tolist()

                    vehicle_results.append((osm_nodes, travel_time))

                remapped_routes.append(vehicle_results)
            else:
                remapped_routes.append([])

        return remapped_routes

    def compute_alternatives(self, vehicles: List[Vehicle], k: int, use_origin_speeds: bool = False) -> List[
        Optional[AlternativeRoutes]]:
        # OD_matrix = [
        #     (self.routing_map.osm_to_hdf5_id(v.next_routing_od_nodes[0]),
        #         self.routing_map.osm_to_hdf5_id(v.next_routing_od_nodes[1]))
        #     for v in vehicles
        # ]
        # self.ru.do_alternatives(OD_matrix, k)
        # 1. Create the list of tuples as you did before
        OD_list = [
            (self.routing_map.osm_to_hdf5_id(v.next_routing_od_nodes[0]),
             self.routing_map.osm_to_hdf5_id(v.next_routing_od_nodes[1]))
            for v in vehicles
        ]

        # Deduplicate identical (origin, destination) pairs: vehicles sharing an
        # OD get the same Plateau result, so each unique pair is computed once
        # and fanned out afterwards. Profiling showed the routing query is the
        # dominant per-step cost, so every duplicate skipped is a full query
        # saved.
        unique_pairs = list(dict.fromkeys(OD_list))
        pair_to_slot = {od: i for i, od in enumerate(unique_pairs)}
        if len(unique_pairs) < len(OD_list):
            logger.debug("OD dedupe: %d vehicles -> %d unique pairs",
                         len(OD_list), len(unique_pairs))

        OD_matrix = np.array(unique_pairs, dtype=np.int32)

        self.ru.do_alternatives(OD_matrix, k, use_origin_speeds)

        li = self.ru.get_routes()

        # Routes come back per unique pair (slots 0..U-1); postprocess only
        # needs the slot count from its second argument.
        unique_routes = self.postprocess_flat(li, unique_pairs)

        # Fan out to vehicles. Node lists are copied per vehicle so no two
        # vehicles share a mutable route object.
        remapped_routes = []
        for od in OD_list:
            slot_routes = unique_routes[pair_to_slot[od]]
            remapped_routes.append([(list(nodes), travel_time)
                                    for nodes, travel_time in slot_routes])
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
        super().__init__()
        self.vehicle_behaviour = VehicleRouteSelection.FIRST

    def select_routes(self, route_possibilities: List[VehicleWithPlans]) -> List[VehicleWithRoute]:
        return [(vehicle, routes[0]) for (vehicle, routes) in route_possibilities]


class RandomRouteSelection(RouteSelectionProvider):

    def __init__(self, seed: int = None):
        super().__init__()
        self.vehicle_behaviour = VehicleRouteSelection.RANDOM
        self.generator = random.Random(seed)


    def select_routes(self, route_possibilities: List[VehicleWithPlans]) -> List[VehicleWithRoute]:
        result = []
        for (vehicle, routes) in route_possibilities:
            route = self.generator.choice(routes)
            result.append((vehicle, route))
        return result


class ZeroMQDistributedPTDRRouteSelection(RouteSelectionProvider):
    """
    Sends routes to a distributed node that calculates a Monte Carlo simulation and returns
    the shortest route for each car.
    """
    def __init__(self, client):
        super().__init__()
        self.vehicle_behaviour = VehicleRouteSelection.PTDR
        self.client = client
        self.ptdr_info = None

    def update_segment_profiles(self, ptdr_info: PTDRInfo):
        self.ptdr_info = ptdr_info
        # self.client.broadcast(Message(kind="load-profiles", data=str(path)))


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
