from __future__ import annotations

import logging
from functools import cache

import networkx as nx
import numpy as np

from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from enum import Enum, unique

from matplotlib.colors import ListedColormap, BoundaryNorm

from .plot import reshape, get_node_coordinates, WidthStyle
from .zoom import get_zoom_level, ZoomLevel


segment_speed_thresholds = [0, 20 / 3.6, 40 / 3.6, 60 / 3.6]


def plot_routes(g: nx.MultiDiGraph,
                ax: Axes,
                nodes_from: list[int],
                nodes_to: list[int],
                densities: list[int] | list[list[int]],
                speeds: list[float] | list[list[float]] | None,
                min_density: int = 1, max_density: int = 10,
                speeds_thresholds: list[float] = None,
                min_width_density: int = 10, max_width_density: int = 50,
                default_linewidth: float = 3, width_modifier: float = 1,
                width_style: WidthStyle = WidthStyle.BOXED,
                round_edges: bool = True,
                roadtypes_by_zoom: bool = False, hidden_lines_width=1,
                delete_if_no_speed: bool = False,
                plot: bool = True):
    """
    Plots routes on a matplotlib axes with speed and density information
    WARNING: parameters min_density, max_density and width_style are not used in this version
    :param g: Graph representation of base layer map
    :param ax: ax layer for adding plotted shapes
    :param nodes_from: OSM id defining starting nodes of segments
    :param nodes_to: OSM id defining ending nodes of segments
    :param densities: list of lists defining number of cars for each part of the segment
    :param speeds: list of lists defining speed of cars for each part of the segment (in m/s)
    :param min_density: NOT USED
    :param max_density: NOT USED
    :param speeds_thresholds: list of speed thresholds in m/s for color coding (default: [0, 20/3.6, 40/3.6, 60/3.6])
    :param min_width_density: density defining width change scope
    :param max_width_density: density defining width change scope
    :param default_linewidth: width of the line with min_width_density (in points)
    :param width_modifier: width of the line with max_width_density (in points)
    :param width_style: ONLY BOXED STYLE IS USED
    :param round_edges: if True plot circles at the end of wide segments for smoother connection
    :param plot: if True add collections to Ax
    :param roadtypes_by_zoom: if True filter segments based on the zoom level of ax
    :param hidden_lines_width: width of the hidden lines (in points)
    :param delete_if_no_speed: if True parts of segments with no speed are not plotted
    :return: LineCollection of color segments
    """
    if speeds is None:
        speeds = densities

    if speeds_thresholds is None:
        speeds_thresholds = segment_speed_thresholds

    if len(speeds_thresholds) != 4:
        logging.error("Please provide 4 speed thresholds")
        speeds_thresholds = segment_speed_thresholds

    lines = []
    color_scalars = []
    widths = []
    zoomed_lines = []
    zoomed_color_scalars = []
    false_segments = 0

    if not (len(nodes_from) == len(nodes_to) and len(nodes_to) == len(densities)):
        logging.error("Nodes_from, nodes_to and densities does not have the same length")

    # get zoom level
    zoom_level = get_zoom_level(ax) if roadtypes_by_zoom else None

    for node_from, node_to, density, speed in zip(nodes_from, nodes_to, densities, speeds):
        if type(density) is int:
            density = [density]

        lines_new, widths_new, speeds_new = plot_route(g, node_from, node_to,
                                                       density, speed,
                                                       zoom_level,
                                                       delete_if_no_speed)
        # get geometry data for smaller zoom
        z_lines_new, z_widths_new, z_speeds_new = None, None, None
        if zoom_level and hidden_lines_width != 0 and lines_new is None:
            z_lines_new, z_widths_new, z_speeds_new = plot_route(g, node_from, node_to, density, speed,
                                                                 zoom_level, delete_if_no_speed)

        if lines_new is not None:
            lines.append(lines_new)
            color_scalars.append(speeds_new)
            widths.append(widths_new)
        elif z_lines_new is not None:
            zoomed_lines.append(z_lines_new)
            zoomed_color_scalars.append(z_speeds_new)
        else:
            false_segments += 1

    if false_segments:
        logging.info(f"False segments: {false_segments} from {len(nodes_from)}")

    if not lines:
        return None, None

    color_scalars = np.hstack(color_scalars)
    widths = np.hstack(widths)

    # width in collection
    line_widths = np.interp(widths, [min_width_density, max_width_density],
                            [default_linewidth, default_linewidth + width_modifier])

    # add width for zoomed segments
    if zoomed_color_scalars:
        zoomed_color_scalars = np.hstack(zoomed_color_scalars)
        arr = np.full(len(zoomed_color_scalars), hidden_lines_width)
        line_widths = np.concatenate((line_widths, arr))
        color_scalars = np.concatenate((color_scalars, zoomed_color_scalars))

    # create collection
    lines.extend(zoomed_lines)
    lines = np.vstack(lines)
    cmap = get_cmap_speeds()
    norm = BoundaryNorm(speeds_thresholds, cmap.N)
    coll = LineCollection(lines, cmap=cmap, norm=norm)

    coll.set_linewidth(line_widths)
    coll.set_array(color_scalars)

    if round_edges:
        coll.set_capstyle('round')

    if plot:
        ax.add_collection(coll, autolim=False)

    return coll, None


def plot_route(g: nx.MultiDiGraph,
               node_from: int,
               node_to: int,
               densities: list[int],
               speeds: list[float],
               zoom_level: ZoomLevel = None,
               delete_if_no_speeds: bool = False):
    x, y = get_node_coordinates(g, node_from, node_to, zoom_level)
    if not x or not y:
        return None, None, None

    if not delete_if_no_speeds:
        speeds = [s for s in speeds if s != -1]

    # color gradient
    line = reshape(x, y)
    density_index = np.interp(np.arange(len(line)), [0, len(line)], [0, len(densities)])
    color_scalar = np.interp(density_index, np.arange(len(densities)), densities)
    speeds_index = np.interp(np.arange(len(line)), [0, len(line)], [0, len(speeds)])
    speed_scalar = np.interp(speeds_index, np.arange(len(speeds)), speeds)

    if delete_if_no_speeds:
        delete_indexes = []
        for i, speed in enumerate(speed_scalar):
            if speed < 0:
                delete_indexes.append(i)

        line = np.delete(line, delete_indexes, axis=0)
        color_scalar = np.delete(color_scalar, delete_indexes)
        speed_scalar = np.delete(speed_scalar, delete_indexes)

    return line, color_scalar, speed_scalar


@cache
def get_cmap_speeds():
    return ListedColormap(['red', 'orange', 'green'])
