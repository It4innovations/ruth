import osmnx as ox
from probduration import Segment

from .data.map import Map
from .data.border import Border, BorderType, PolygonBorderDef

def get_map(polygon: str,
            kind: BorderType,
            name=None,
            on_disk=False,
            with_speeds=False,
            data_dir="./data",
            load_from_cache=True):

    """Get map based on polygon."""
    border_def = PolygonBorderDef(polygon, on_disk=on_disk)
    border_kind = BorderType.parse(kind)
    name_ = name if name is not None else f"custom_{boder_def.md5()}"
    border = Border(name, border_def, border_kind, data_dir, load_from_cache)

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
