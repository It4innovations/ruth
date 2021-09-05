"""Car module."""

from datetime import timedelta
from dataclasses import dataclass, field, InitVar
from typing import List
import csv

import osmnx as ox
from probduration import Segment, SegmentPosition

from .data.geopoint import GeoPoint
from .data.map import Map
from .data.border import Border

OsmNodeId = int


# TODO: maybe don't use the car but vehicle
@dataclass
class Car:
    """Description of the car."""

    id: int
    departure_time_offset: timedelta
    step: timedelta
    starting_point: GeoPoint
    destination_point: GeoPoint
    route_area: InitVar[Border]

    routing_map: Map = field(init=False)
    osm_route: List[int] = field(init=False)

    _segment_position: SegmentPosition = SegmentPosition(0, 0.0)

    def __post_init__(self, route_area):
        """Initialize the unitialized fields."""
        start_ = self.starting_point.point()
        end_ = self.destination_point.point()

        self.routing_map = self._load_map(route_area)

        starting_node = ox.distance.nearest_nodes(
            self.routing_map.network, start_.x, start_.y)
        destination_node = ox.distance.nearest_nodes(
            self.routing_map.network, end_.x, end_.y)

        self.osm_route = [starting_node, destination_node]

    def _load_map(self, route_area):

        routing_map = Map(route_area)
        # extend the map network about speeds on edge
        routing_map.network = ox.add_edge_speeds(routing_map.network)
        return routing_map

    def _routing_origin_dest_index(self):
        """Compute the next origin/destination for routing.

        NOTE: in case the car already started on the current segment
              the end of the segment cannot be replaned and the only change
              can be done in the "next end" => index + 1.
        """
        sp = self.segment_position
        segment_idx = sp.index if sp.start == 0.0 else sp.index + 1

        assert segment_idx < len(self.osm_route)

        return (self.osm_route[segment_idx], self.osm_route[-1], segment_idx)

    def shortest_path(self):
        """Compute the shortest path from the current position to the end."""
        current_starting_node, destination_node, segment_index = \
            self._routing_origin_dest_index()

        osm_route = self.routing_map.shortest_path(current_starting_node,
                                                   destination_node)

        return self.osm_route[:segment_index] + osm_route

    def k_shortest_path(self, k):
        """Compute k-shortest path from the current position to the end."""
        current_starting_node, destination_node, segment_index = \
            self._routing_origin_dest_index()

        osm_routes = self.routing_map.k_shortest_patshs(
            current_starting_node, destination_node, k)

        return list(map(lambda osm_route: self.osm_route[:segment_index] + osm_route, osm_routes))

    def set_osm_route(self, osm_route):
        """Set the current route."""
        self.osm_route = osm_route

    def route_to_segments(self, osm_route):
        """Prepare list of segments based on route."""
        edge_data = ox.utils_graph.get_route_edge_attributes(self.routing_map.network,
                                                             osm_route)
        edges = zip(osm_route, osm_route[1:])
        return [
            Segment(
                f"OSM{from_}T{to_}",
                data["length"],
                data["speed_kph"],
            )
            # NOTE: the zip is correct as the starting node_id is of the interest
            for i, ((from_, to_), data) in enumerate(zip(edges, edge_data))
        ]

    @property
    def segment_position(self):
        """Access the current segment position of the car."""
        return self._segment_position

    def advance(self, segment_position):
        """Advance the car to a new position on a segment."""
        self._segment_position = segment_position


def load_cars(path, route_area) -> List[Car]:
    """Load the cars from the file."""
    cars = []
    with open(path) as csv_file:
        first_line = csv_file.readline()
        sniffer = csv.Sniffer()

        csv_file.seek(0)  # return to the beginning in the file
        csv_reader = csv.reader(csv_file, delimiter=';')
        if sniffer.has_header(first_line):
            next(csv_reader)  # skip header

        for i, line in enumerate(csv_reader):
            cars.append(
                Car(
                    int(line[0]),
                    timedelta(seconds=int(line[7])),
                    timedelta(seconds=int(line[6])),
                    GeoPoint(float(line[1]), float(line[2])),
                    GeoPoint(float(line[3]), float(line[4])),
                    route_area
                )
            )
    return cars
