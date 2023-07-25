"""The state less implementation of a vehicle."""
import sys
from collections import namedtuple
from dataclasses import InitVar, asdict, dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import pandas as pd
from networkx.exception import NodeNotFound

from .data.map import Map
from .simulator.segment import Route, SegmentPosition
from .utils import get_map, round_timedelta


def set_numpy_type(name, fld=None):
    if fld is None:
        fld = field()
    fld.metadata = {'numpy_type': name}
    return fld


IndexedNode = namedtuple("IndexedNode", ["node", "index"])


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
    border_id: str = set_numpy_type("string")
    border: str = set_numpy_type("string")  # polygon definition
    border_kind: str = set_numpy_type("string")
    osm_route: List[int] = set_numpy_type("object")
    active: bool = set_numpy_type("bool")
    """A period in wicht the raw FCD data are sampled"""
    fcd_sampling_period: timedelta = set_numpy_type("object")
    """A history of last several steps."""
    leap_history: List[Tuple[datetime, str, float, float, float, str]] = set_numpy_type(
        "object")  # TODO: is it just a _leap_, isn't it the entire history? Maybe rename to raw_fcd
    #       well there are two things: 1) after each leap I need to make an aggregation and update global view
    #                                  2) collect raw fcds for latter aggregation and creating the prob profiles
    status: str = set_numpy_type("string")
    routing_map: InitVar[Map] = None

    def __post_init__(self, routing_map):
        # NOTE: the routing map is not among attributes of dataclass
        # => does not affect the conversion to pandas.Series
        if routing_map is None:
            self.routing_map = get_map(self.border, self.border_kind,
                                       with_speeds=True, name=self.border_id)
        else:
            self.routing_map = routing_map

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
        self.__post_init__(None)

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

    def shortest_path(self):
        """Compute the shortest path from the current position to the end."""
        current_starting_node = self.next_routing_start.node

        return self.routing_map.shortest_path(current_starting_node, self.dest_node)

    def k_shortest_paths(self, k: int) -> Optional[List[List[int]]]:
        """Compute k-shortest path from the current position to the end."""
        current_starting_node = self.next_routing_start.node

        if self.osm_route is None:
            # No route between origin and destination
            return None

        try:
            osm_routes = self.routing_map.k_shortest_paths(current_starting_node, self.dest_node, k)
            return list(osm_routes)
        except NodeNotFound as ex:
            print(f"vehicle: {self.id}: {ex}", file=sys.stderr)
            return None

    def k_fastest_paths(self, k: int) -> Optional[List[List[int]]]:
        """Compute k-fastest path from the current position to the end."""
        current_starting_node = self.next_routing_start.node

        if self.osm_route is None:
            # No route between origin and destination
            return None

        try:
            osm_routes = self.routing_map.k_fastest_paths(current_starting_node, self.dest_node, k)
            return list(osm_routes)
        except NodeNotFound as ex:
            print(f"vehicle: {self.id}: {ex}", file=sys.stderr)
            return None

    def update_followup_route(self, osm_route: Route):
        """
        Updates the route from the current node to the destination node.
        """
        assert osm_route[-1] == self.dest_node
        node_index = self.next_routing_start.index
        self.osm_route = self.osm_route[:node_index] + osm_route

    @property
    def segment_position(self) -> SegmentPosition:
        return SegmentPosition(index=self.start_index, position=self.start_distance_offset)

    def set_position(self, sp: SegmentPosition):
        self.start_index = sp.index
        self.start_distance_offset = sp.position

    def store_fcd(self, start_offset, duration, segment, pos_start, speed_mps):
        step_m = speed_mps * (self.fcd_sampling_period / timedelta(seconds=1))

        start = pos_start
        current_offset = start_offset
        end_offset = start_offset + duration
        while current_offset + self.fcd_sampling_period < end_offset and \
                start + step_m < segment.length:
            start += step_m
            current_offset += self.fcd_sampling_period
            self.leap_history.append(
                (current_offset, segment.id, start, speed_mps, segment.length, self.status))

        step_m = speed_mps * ((end_offset - current_offset) / timedelta(seconds=1))
        if start + step_m < segment.length:
            # TODO: the question is wheather to store all the cars at the end of period or
            # rather return the difference (end_offset - _last_ current_offset) and take it as
            # a parameter for the next round of storing. In this way all the cars would be sampled
            # with an exact step (car dependent as each car can have its own sampling period)
            self.leap_history.append(
                (end_offset, segment.id, start + step_m, speed_mps, segment.length, self.status))

    def is_active(self, within_offset, freq):
        return self.active and within_offset == round_timedelta(self.time_offset, freq)

    def __repr__(self):
        return f"Vehicle(id={self.id}, active={self.active}, pos={self.segment_position})"
