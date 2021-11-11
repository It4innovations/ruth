"""The state less implementation of a vehicle."""

import pandas as pd

from dataclasses import dataclass, field, asdict, is_dataclass
from typing import List
from datetime import timedelta
from probduration import SegmentPosition

from .utils import get_map


@dataclass
class Vehicle:
    """Vehicle."""

    id: int
    time_offset: timedelta
    frequency: timedelta
    start_index: int
    start_distance_offset: float
    origin_node: int
    dest_node: int
    border_id: str
    osm_route: List[int]
    active: bool

    def __post_init__(self):
        # NOTE: the routing map is not among attributes of dataclass
        # => does not affect the conversion to pandas.Series
        self.routing_map = get_map(self.border_id, with_speeds=True)

        if not self.osm_route:  # empty route
            # NOTE: set dummy route consisted only from the origin and destination nodes.
            # At the beginning the _start_index_ and _starting_distance_offset_ are 0 and 0.0,
            # hence the *current starting node* with *index* will be the _origin_node_ and 0.
            self.osm_route = [self.origin_node, self.dest_node]

    @property
    def next_routing_start_node_with_index(self):
        """Compute the next origin for routing.

        NOTE: in case the car already started on the current segment
              the end of the segment cannot be replaned and the only change
              can be done in the "next end" => index + 1.
        """
        segment_idx = (self.start_index
                       if self.start_distance_offset == 0.0
                       else self.start_index + 1)
        assert segment_idx < len(self.osm_route)

        return (self.osm_route[segment_idx], segment_idx)

    @property
    def current_node(self):
        """Return the actual node which segment is in processing."""
        return self.osm_route[self.start_index]

    def shortest_path(self):
        """Compute the shortest path from the current position to the end."""
        current_starting_node, segment_index = \
            self.next_routing_start_node_with_index

        osm_route = self.routing_map.shortest_path(current_starting_node,
                                                   self.dest_node)

        return self.osm_route[:segment_index] + osm_route

    def k_shortest_paths(self, k):
        """Compute k-shortest path from the current position to the end."""
        current_starting_node, segment_index = \
            self.next_routing_start_node_with_index

        osm_routes = self.routing_map.k_shortest_patshs(
            current_starting_node, self.dest_node, k)

        return list(map(lambda osm_route: self.osm_route[:segment_index] + osm_route, osm_routes))

    def set_current_route(self, osm_route):
        """Set the current route."""
        self.osm_route = osm_route

    @property
    def segment_position(self):
        return SegmentPosition(self.start_index, self.start_distance_offset)

    def set_position(self, sp: SegmentPosition):
        self.start_index = sp.index
        self.start_distance_offset = sp.start
