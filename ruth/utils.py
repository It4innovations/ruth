import osmnx as ox

from probduration import Segment
from .data.map import Map
from .data.cz import Boundary

boundary = Boundary()  # NOTE: ok as loading is lazy


def get_map(border_id, with_speeds=False):
    """Get map based on border id."""
    border = getattr(boundary, border_id)
    return Map(border, with_speeds=with_speeds)


def osm_route_to_segments(osm_route, routing_map):
    """Prepare list of segments based on route."""
    edge_data = ox.utils_graph.get_route_edge_attributes(routing_map.network,
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
