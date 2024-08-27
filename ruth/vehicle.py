import sys
from collections import namedtuple
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from enum import Enum, auto
from math import isclose
from typing import Any, List, Optional, Tuple

import random
import pandas as pd
from networkx.exception import NodeNotFound

from .data.map import Map
from .data.segment import Route, SegmentPosition, LengthMeters, RouteWithTime
from .utils import round_timedelta


def set_numpy_type(name, fld=None):
    if fld is None:
        fld = field()
    fld.metadata = {'numpy_type': name}
    return fld


IndexedNode = namedtuple("IndexedNode", ["node", "index"])


class VehicleAlternatives(Enum):
    """
    Enum for different types of route alternatives calculation.

    DEFAULT: No alternatives are calculated. The vehicle sticks to its precalculated route taken from the input
    parquet file.
    DIJKSTRA_FASTEST: The alternatives are calculated using networkx's k_shortest_paths function with the Dijkstra
    algorithm. Travel time is set as the edge weight.
    DIJKSTRA_SHORTEST: The alternatives are calculated using networkx's k_shortest_paths function with the Dijkstra
    algorithm. Segment length is set as the edge weight.
    PLATEAU_FASTEST: The alternatives are calculated using the plateau algorithm implemented in evkit
    (https://code.it4i.cz/everest/evkit). Travel time is set as the edge weight.
    """

    DEFAULT = 0
    DIJKSTRA_FASTEST = auto()
    DIJKSTRA_SHORTEST = auto()
    PLATEAU_FASTEST = auto()


class VehicleRouteSelection(Enum):
    """
    Enum for different types of route selection from the calculated alternatives.

    NO_ALTERNATIVE: No alternative is selected. Set if DEFAULT is set in VehicleAlternatives.
    FIRST: The first alternative is selected.
    RANDOM: A random alternative is selected from the calculated alternatives.
    PTDR: The alternative is selected using the probability profiles. Implemented in evkit.
    """
    NO_ALTERNATIVE = 0
    FIRST = auto()
    RANDOM = auto()
    PTDR = auto()


def set_vehicle_behavior(vehicles: List['Vehicle'],
                         alternatives_ratio: List[float],
                         route_selection_ratio: List[float]):
    """
    Selects random vehicles according to the ratio set in parameters and sets their type of alternatives calculation
    and route selection from the calculated alternatives.
    """
    assert isclose(sum(alternatives_ratio), 1, abs_tol=1e-8)
    assert isclose(sum(route_selection_ratio), 1, abs_tol=1e-8)
    if alternatives_ratio[0] != route_selection_ratio[0]:
        # The percentage of DEFAULT alternatives calculation must be equal to the percentage of NO_ALTERNATIVE route
        # selection.
        raise ValueError("Sum of percentages for the alternatives calculation and route selection set in parameters "
                         "must be equal.")

    n_vehicles = len(vehicles)
    vehicles_shuffled = list(vehicles)
    random.shuffle(vehicles_shuffled)

    # select the number of vehicles to change
    n_vehicles_to_change_alt = [int(r * n_vehicles) for r in alternatives_ratio]
    sum_n_vehicles_to_change = sum(n_vehicles_to_change_alt)
    if sum_n_vehicles_to_change != n_vehicles:
        # add the difference to the DEFAULT
        n_vehicles_to_change_alt[0] += n_vehicles - sum_n_vehicles_to_change

    n_vehicles_to_change_selection = [int(r * n_vehicles) for r in route_selection_ratio]
    n_vehicles_to_change_selection = n_vehicles_to_change_alt[0:1] + n_vehicles_to_change_selection[1:]

    index_from = 0
    for n, alternative in zip(n_vehicles_to_change_alt, VehicleAlternatives):
        index_to = index_from + n
        for v in vehicles_shuffled[index_from:index_to]:
            v.alternatives = alternative
        index_from = index_from + n

    index_from = n_vehicles_to_change_alt[0]
    vehicles_with_alternatives = vehicles_shuffled[index_from:]
    vehicles_shuffled = (vehicles_shuffled[:index_from]
                         + random.sample(vehicles_with_alternatives, len(vehicles_with_alternatives)))
    n_vehicles_to_change = [int(r * n_vehicles) for r in route_selection_ratio]

    index_from = 0
    for n, route_selection in zip(n_vehicles_to_change_selection, VehicleRouteSelection):
        index_to = index_from + n
        for v in vehicles_shuffled[index_from:index_to]:
            v.route_selection = route_selection
        index_from = index_from + n

    return


@dataclass
class Vehicle:
    """Vehicle."""

    id: int = set_numpy_type("int64")
    time_offset: timedelta = set_numpy_type("object")
    frequency: timedelta = set_numpy_type("object")
    start_index: int = set_numpy_type("int64")
    start_distance_offset: float = set_numpy_type("float64")
    origin_node: int = set_numpy_type("int64")
    dest_node: int = set_numpy_type("int64")
    osm_route: List[int] = set_numpy_type("object")
    active: bool = set_numpy_type("bool")
    """A period in which the raw FCD data are sampled"""
    fcd_sampling_period: timedelta = set_numpy_type("object")
    status: str = set_numpy_type("string")
    alternatives: VehicleAlternatives = VehicleAlternatives.DEFAULT
    route_selection: VehicleRouteSelection = VehicleRouteSelection.NO_ALTERNATIVE
    # Kept for backwards compatibility with old input files
    leap_history: Any = None

    def __post_init__(self):
        # We want to normalize these values to datetime.timedelta, because performing operations
        # between pandas.TimeDelta and datetime.timedelta is very slow (10x slower).
        # The check is here because the values are pandas when initially loaded from disk,
        # but then they are already converted to datetime when they are sent between processes
        # and reinitialized.
        if isinstance(self.frequency, pd.Timedelta):
            self.frequency = self.frequency.to_pytimedelta()
        if isinstance(self.time_offset, pd.Timedelta):
            self.time_offset = self.time_offset.to_pytimedelta()

    def __getstate__(self):
        state = asdict(self)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__post_init__()

    def __eq__(self, other: 'Vehicle'):
        return self.id == other.id

    @property
    def next_routing_od_nodes(self) -> Tuple[int, int]:
        return self.next_routing_start.node, self.dest_node

    @property
    def next_routing_start(self) -> IndexedNode:
        """Compute the next origin for routing.

        NOTE: in case the car already started on the current segment
              the end of the segment cannot be re-planed and the only change
              can be done in the "next end" => index + 1.
        """
        node_idx = (self.start_index
                    if self.start_distance_offset == 0.0
                    else self.start_index + 1)
        assert node_idx < len(self.osm_route)

        return IndexedNode(self.osm_route[node_idx], node_idx)

    @property
    def current_node(self):
        """Return the actual node which segment is in processing."""
        return self.osm_route[self.start_index]

    @property
    def next_node(self):
        """Return the end node of segment in processing."""
        return self.osm_route[self.start_index + 1]

    @property
    def map_id(self) -> int:
        return self.current_travel_time.map_id if self.current_travel_time is not None else -1

    def shortest_path(self, routing_map: Map) -> Optional[List[int]]:
        """Compute the shortest path from the current position to the end."""
        current_starting_node = self.next_routing_start.node

        return routing_map.shortest_path(current_starting_node, self.dest_node)

    def k_shortest_paths(self, k: int, routing_map: Map) -> Optional[List[List[int]]]:
        """Compute k-shortest path from the current position to the end."""
        current_starting_node = self.next_routing_start.node

        if self.osm_route is None:
            # No route between origin and destination
            return None

        try:
            osm_routes = routing_map.k_shortest_paths(current_starting_node, self.dest_node, k)
            return list(osm_routes)
        except NodeNotFound as ex:
            print(f"vehicle: {self.id}: {ex}", file=sys.stderr)
            return None

    def k_fastest_paths(self, k: int, routing_map: Map) -> Optional[List[List[int]]]:
        """Compute k-fastest path from the current position to the end."""
        current_starting_node = self.next_routing_start.node

        if self.osm_route is None:
            # No route between origin and destination
            return None

        try:
            osm_routes = routing_map.k_fastest_paths(current_starting_node, self.dest_node, k)
            return list(osm_routes)
        except NodeNotFound as ex:
            print(f"vehicle: {self.id}: {ex}", file=sys.stderr)
            return None

    def has_next_segment_closed(self, routing_map: Map) -> bool:
        assert self.start_index + 2 < len(self.osm_route)  # 2 more nodes ahead to make a segment
        next_segment_from, next_segment_to = self.osm_route[self.start_index + 1], self.osm_route[self.start_index + 2]
        max_speed_on_next_segment = routing_map.get_current_max_speed(next_segment_from, next_segment_to)
        return max_speed_on_next_segment == 0.0

    def update_followup_route(self, suggested_route_with_time: RouteWithTime, routing_map: Map, travel_time_limit_perc: float = None):
        """
        Updates the route from the current node to the destination node.
        """
        suggested_route, suggested_route_travel_time = suggested_route_with_time
        assert suggested_route[-1] == self.dest_node
        node_index = self.next_routing_start.index
        first_part = self.osm_route[:node_index]
        if len(first_part) > 0:
            assert first_part[-1] != suggested_route[0]

        current_route = self.osm_route[node_index:]

        if travel_time_limit_perc is None or self.alternatives == VehicleAlternatives.DIJKSTRA_SHORTEST:
            self.osm_route = first_part + suggested_route
            return

        current_route_travel_time = routing_map.get_path_travel_time(current_route)
        travel_time_limit = current_route_travel_time * (1 - travel_time_limit_perc)

        if suggested_route_travel_time is None:
            if routing_map.check_if_travel_time_is_faster(suggested_route, travel_time_limit):
                self.osm_route = first_part + suggested_route
            return

        elif suggested_route_travel_time < travel_time_limit:
            self.osm_route = first_part + suggested_route


    @property
    def segment_position(self) -> SegmentPosition:
        return SegmentPosition(index=self.start_index, position=LengthMeters(self.start_distance_offset))

    def set_position(self, sp: SegmentPosition):
        self.start_index = sp.index
        self.start_distance_offset = sp.position

    def is_active(self, within_offset, freq):
        return self.active and within_offset == round_timedelta(self.time_offset, freq)

    def is_at_the_end_of_segment(self, routing_map: Map):
        node_from, node_to = self.osm_route[self.start_index], self.osm_route[self.start_index + 1]
        osm_segment = routing_map.get_osm_segment(node_from, node_to)
        return self.start_distance_offset == osm_segment.length

    def __repr__(self):
        return f"Vehicle(id={self.id}, active={self.active}, pos={self.segment_position})"
