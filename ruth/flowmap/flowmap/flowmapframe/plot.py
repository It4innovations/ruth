from __future__ import annotations

import logging
from functools import cache

import networkx as nx
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PatchCollection
from enum import Enum, unique

from matplotlib.colors import ListedColormap

from .preprocessing import get_width_polygon
from .zoom import get_zoom_level, get_highway_types, ZoomLevel


@unique
class WidthStyle(Enum):
    NONE = 0
    """
    uses only color
    """
    BOXED = 1
    """
    uses matplotlib.LineCollection
    """
    CALLIGRAPHY = 2
    """
    uses matplotlib.fill_between / matplotlib.fill_betweenx
    """
    EQUIDISTANT = 3
    """
    uses matplotlib.PatchCollection
    """


def plot_routes(g: nx.MultiDiGraph,
                ax: Axes,
                nodes_from: list[int],
                nodes_to: list[int],
                densities: list[int] | list[list[int]],
                min_density: int = 1, max_density: int = 10,
                min_width_density: int = 10, max_width_density: int = 50,
                default_linewidth: float = 3, width_modifier: float = 1,
                width_style: WidthStyle = WidthStyle.BOXED,
                round_edges: bool = True,
                roadtypes_by_zoom: bool = False, hidden_lines_width = 1,
                plot: bool = True):
    """
    Plotting of segments into ax with their density represented by color and width
    :param g: Graph representation of base layer map
    :param ax: layer for adding plotted shapes
    :param nodes_from: OSM id defining starting nodes of segments
    :param nodes_to: OSM id defining ending nodes of segments
    :param densities: list of lists defining number of cars for each part of the segment
    :param min_density: density defining color gradient scope
    :param max_density: density defining color gradient scope
    :param min_width_density: density defining width change scope
    :param max_width_density: density defining width change scope
    :param default_linewidth: width of the line with min_width_density (in points)
    :param width_modifier: width of the line with max_width_density (in points)
    :param width_style: style of the width representation
    :param round_edges: if True plot circles at the end of wide segments for smoother connection
    :param plot: if True add collections to Ax
    :param roadtypes_by_zoom: if True filter segments based on the zoom level of ax
    :return: LineCollection of color segments, PatchCollection of width representation
    """
    lines = []
    color_scalars = []
    polygons = []
    zoomed_lines = []
    zoomed_color_scalars = []
    false_segments = 0

    if not (len(nodes_from) == len(nodes_to) and len(nodes_to) == len(densities)):
        logging.error(f"Nodes_from, nodes_to and densities does not have the same length")

    # get zoom level
    zoom_level = get_zoom_level(ax) if roadtypes_by_zoom else None

    for node_from, node_to, density in zip(nodes_from, nodes_to, densities):
        if type(density) is int:
            density = [density]

        lines_new, color_scalars_new, polygons_new = plot_route(g, node_from, node_to, density, ax,
                                                                min_width_density, max_width_density,
                                                                width_modifier=width_modifier,
                                                                width_style=width_style, round_edges=round_edges,
                                                                zoom_level=zoom_level)
        # get geometry data for smaller zoom
        z_lines_new, z_colors_new = None, None
        if zoom_level and hidden_lines_width != 0 and lines_new is None:
            z_lines_new, z_colors_new, _ = plot_route(g, node_from, node_to, density, ax,
                                                      min_width_density, max_width_density,
                                                      width_modifier=width_modifier,
                                                      width_style=None,
                                                      zoom_level=None)

        if lines_new is not None:
            lines.append(lines_new)
            color_scalars.append(color_scalars_new)
            polygons.extend(polygons_new)
        elif z_lines_new is not None:
            zoomed_lines.append(z_lines_new)
            zoomed_color_scalars.append(z_colors_new)
        else:
            false_segments += 1

    if false_segments:
        logging.info(f"False segments: {false_segments} from {len(nodes_from)}")

    if not lines:
        return None, None

    color_scalars = np.hstack(color_scalars)

    # width in collection
    if width_style == WidthStyle.BOXED:
        line_widths = np.interp(color_scalars, [min_width_density, max_width_density],
                                [default_linewidth, default_linewidth + width_modifier])
    else:
        line_widths = np.full(len(color_scalars), default_linewidth)

    # add width for zoomed segments
    if zoomed_color_scalars:
        zoomed_color_scalars = np.hstack(zoomed_color_scalars)
        arr = np.full(len(zoomed_color_scalars), hidden_lines_width)
        line_widths = np.concatenate((line_widths, arr))
        color_scalars = np.concatenate((color_scalars, zoomed_color_scalars))

    # create collection
    lines.extend(zoomed_lines)
    lines = np.vstack(lines)
    cmap, norm = get_color_bar_info(min_density, max_density)
    coll = LineCollection(lines, cmap=cmap, norm=norm)

    coll.set_linewidth(line_widths)
    coll.set_array(color_scalars)

    if round_edges:
        coll.set_capstyle('round')

    if plot:
        ax.add_collection(coll, autolim=False)

    patch = None
    if polygons:
        patch = PatchCollection(polygons)
        patch.set_facecolor(get_cmap()(1.0))
        if plot:
            ax.add_collection(patch, autolim=False)

    return coll, patch


def plot_route(g: nx.MultiDiGraph,
               node_from: int,
               node_to: int,
               densities: list[int],
               ax: Axes,
               min_width_density: int,
               max_width_density: int,
               width_modifier: float,
               width_style: WidthStyle | None,
               round_edges: bool = True,
               zoom_level: ZoomLevel = None):
    x, y = get_node_coordinates(g, node_from, node_to, zoom_level)
    if not x or not y:
        return None, None, None

    # edit length of densities to match length of x
    a = np.arange(len(x))
    density_index = np.interp(a, [0, len(x)], [0, len(densities)])
    point_densities = np.interp(density_index, np.arange(len(densities)), densities)

    # color gradient
    line = reshape(x, y)
    density_index = np.interp(np.arange(len(line)), [0, len(line)], [0, len(densities)])
    color_scalar = np.interp(density_index, np.arange(len(densities)), densities)

    # width as filling
    polygons = []
    if width_style == WidthStyle.CALLIGRAPHY:
        polygons = get_width_polygon(ax, x, y, point_densities, min_width_density, max_width_density,
                                     width_modifier, equidistant=False, round_edges=round_edges)

    elif width_style == WidthStyle.EQUIDISTANT:
        polygons = get_width_polygon(ax, x, y, point_densities, min_width_density, max_width_density,
                                     width_modifier, equidistant=True, round_edges=round_edges)

    return line, color_scalar, polygons


def get_node_coordinates(g, node_from, node_to, zoom_level=None):
    x = []
    y = []
    edge = g.get_edge_data(node_from, node_to)
    if edge is None:
        return None, None

    data = min(edge.values(), key=lambda d: d["length"])
    if 'highway' in data and zoom_level is not None:
        if data['highway'] not in get_highway_types(zoom_level):
            return None, None

    if "geometry" in data:
        xs, ys = data["geometry"].xy
        x.extend(xs)
        y.extend(ys)
    else:
        x.extend((g.nodes[node_from]["x"], g.nodes[node_to]["x"]))
        y.extend((g.nodes[node_from]["y"], g.nodes[node_to]["y"]))

    return x, y


def reshape(x, y):
    """
    Transforms x and y coords lists into list of lines (defined by START and END point)
    """
    points = np.vstack([x, y]).T.reshape(-1, 1, 2)
    points = np.concatenate([points[:-1], points[1:]], axis=1)
    return points


@cache
def get_cmap():
    cmap = plt.get_cmap('autumn_r', 512)
    newcmp = ListedColormap(cmap(np.linspace(0.25, 1, 256)))
    return newcmp

@cache
def get_color_bar_info(min_density: int, max_density: int):
    cmap = get_cmap()
    norm = plt.Normalize(min_density, max_density)
    return cmap, norm

