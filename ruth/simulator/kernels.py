from typing import List, Optional, Tuple
import random

import osmnx as ox
from matplotlib import pyplot as plt
from osmnx.plot import get_colors

from .segment import Route
from ..data.map import Map, MapProvider
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
    def __init__(self, port: int):
        from ..zeromq.src.client import Client
        self.client = Client(port=port)

    def load_map(self, map: Map):
        """
        Loads updated information from the passed map.
        """
        # TODO: implement with broadcast
        self.client.compute([Message(kind="load-map", data=map.hdf5_file_path)])

    def compute_alternatives(self, map: Map, vehicles: List[Vehicle], k: int) -> List[
        Optional[AlternativeRoutes]]:
        messages = [Message(kind="alternatives", data={
            "start": map.osm_to_hdf5_id(v.next_routing_od_nodes[0]),
            "destination": map.osm_to_hdf5_id(v.next_routing_od_nodes[1]),
            "max_routes": k
        }) for v in vehicles]
        # logging.info(f"Sending alternatives to distributed worker: {messages}")
        results = self.client.compute(messages)
        # logging.info(f"Response from worker: {results}")
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

    def select_routes(self, route_possibilities: List[VehicleWithPlans]) -> List[VehicleWithRoute]:
        raise NotImplementedError


def onclick(event, graph):
    x, y = event.xdata, event.ydata
    node = ox.nearest_nodes(graph, x, y, return_dist=False)
    print(node)


def plot_alternatives(route_possibilities):
    g = MapProvider.get_map()

    # ec = ['blue' if (u == 25664661 and v == 27349583) else 'black' for u, v, k in g.edges(keys=True)]
    # fig, ax = ox.plot_graph(g, node_color='w', node_edgecolor='k', node_size=2, bgcolor="white",
    #                         node_zorder=3, edge_color=ec, edge_linewidth=0.5, show=False)

    for vehicle, routes in route_possibilities:
        colors = get_colors(len(routes), cmap="gist_rainbow", return_hex=True)
        print(len(routes))
        if not routes:
            return
        elif len(routes) == 1:
            fig, ax = ox.plot_graph_route(g, routes[0], route_color=colors[0], bgcolor="white", edge_color="black",
                                          edge_linewidth=0.3, show=False, close=False)
        else:
            fig, ax = ox.plot_graph_routes(g, routes, route_colors=colors, route_linewidth=0.5, node_size=0,
                                           node_color='black', bgcolor="white", edge_color="black", edge_linewidth=0.3,
                                           show=False, close=False)
        cid = fig.canvas.mpl_connect('button_press_event', lambda e: onclick(e, g))
        plt.show()


class FirstRouteSelection(RouteSelectionProvider):
    """
    Selects the first route for each car.
    """

    def select_routes(self, route_possibilities: List[VehicleWithPlans]) -> List[VehicleWithRoute]:
        return [(vehicle, routes[0]) for (vehicle, routes) in route_possibilities]


class RandomRouteSelection(RouteSelectionProvider):

    def __init__(self):
        # TODO: add seed from config
        random.seed(1)

    """
    Selects random route for each car.
    """

    def select_routes(self, route_possibilities: List[VehicleWithPlans]) -> List[VehicleWithRoute]:
        result = []
        # plot_alternatives(route_possibilities)
        for (vehicle, routes) in route_possibilities:
            index = random.randrange(len(routes))
            result.append((vehicle, routes[index]))
        return result
