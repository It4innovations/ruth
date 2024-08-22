import networkx as nx
import osmnx as ox

from enum import Enum, unique
from matplotlib import pyplot as plt
from matplotlib.axes import Axes


@unique
class ZoomLevel(Enum):
    LEVEL_ONE = 320_000
    LEVEL_TWO = 161_000
    LEVEL_THREE = 40_000
    LEVEL_FOUR = 0

    def get_smaller_zooms(self):
        if self == ZoomLevel.LEVEL_FOUR:
            return []
        elif self == ZoomLevel.LEVEL_THREE:
            return [ZoomLevel.LEVEL_FOUR]
        elif self == ZoomLevel.LEVEL_TWO:
            return [ZoomLevel.LEVEL_THREE, ZoomLevel.LEVEL_FOUR]
        elif self == ZoomLevel.LEVEL_ONE:
            return [ZoomLevel.LEVEL_TWO, ZoomLevel.LEVEL_THREE, ZoomLevel.LEVEL_FOUR]
        return []


def get_zoom_level(ax) -> ZoomLevel:
    lims = ax.get_xlim()
    lims = lims[1] - lims[0]
    window_size = ax.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted()).size
    ratio = lims / window_size[0]
    ratio = ratio * 111 * 39370.0787
    for zoom_level in ZoomLevel:
        if ratio > zoom_level.value:
            return zoom_level


def get_highway_types(zoom_level: ZoomLevel):
    if zoom_level == ZoomLevel.LEVEL_ONE:
        return ['motorway', 'motorway_link'
                            'primary', 'primary_link',
                'trunk', 'trunk_link']
    if zoom_level == ZoomLevel.LEVEL_TWO:
        return ['motorway', 'motorway_link',
                'trunk', 'trunk_link'
                'primary', 'primary_link',
                'secondary', 'secondary_link']
    if zoom_level == ZoomLevel.LEVEL_THREE:
        return ['motorway', 'motorway_link',
                'trunk', 'trunk_link',
                'primary', 'primary_link',
                'secondary', 'secondary_link',
                'tertiary', 'tertiary_link']
    if zoom_level == ZoomLevel.LEVEL_FOUR:
        return ['motorway', 'motorway_link',
                'trunk', 'trunk_link',
                'primary', 'primary_link',
                'secondary', 'secondary_link'
                'tertiary', 'tertiary_link',
                'unclassified', 'residential',
                'service', 'living_street', 'road', 'track'
                ]
    return []


def plot_graph_with_zoom(g: nx.MultiDiGraph,
                         ax: Axes,
                         color_primary="dimgray",
                         size_primary: float = 1,
                         secondary_colors: list = None,
                         secondary_sizes: list = None):
    # check if secondary colors and sizes are correct length
    if secondary_sizes is None:
        secondary_sizes = [1, 0.7, 0.5]
    if secondary_colors is None:
        secondary_colors = ["darkgray"]

    if len(secondary_colors) < len(ZoomLevel):
        add = [secondary_colors[-1] for _ in range(len(ZoomLevel) - len(secondary_colors))]
        secondary_colors.extend(add)

    if len(secondary_sizes) < len(ZoomLevel):
        add = [secondary_sizes[-1] for _ in range(len(ZoomLevel) - len(secondary_sizes))]
        secondary_sizes.extend(add)

    # check if graph has already been plotted
    if ax is None:
        ax = plt.gca()

    lines = ax.collections
    if not lines:
        _, ax = ox.plot_graph(g, ax=ax, node_size=0, show=False)

    lines = ax.collections
    if not lines:
        return ax

    # get zoom level
    zoom_level = get_zoom_level(ax)

    if not zoom_level:
        ax.collections[0].set_color(color_primary)
        ax.collections[0].set_linewidth(size_primary)
        return ax

    # get colors and sizes for each edge in graph
    e_colors = []
    e_sizes = []
    for u, v, k, d in g.edges(keys=True, data=True):
        if type(d['highway']) is list:
            d['highway'] = d['highway'][0]

        if d['highway'] in get_highway_types(zoom_level):
            e_colors.append(color_primary)
            e_sizes.append(size_primary)
        else:
            colored = False
            for i, zoom in enumerate(zoom_level.get_smaller_zooms()):
                if d['highway'] in get_highway_types(zoom):
                    e_colors.append(secondary_colors[i])
                    e_sizes.append(secondary_sizes[i])
                    colored = True
                    break
            if not colored:
                e_colors.append('white')
                e_sizes.append(1)

    ax.collections[0].set_color(e_colors)
    ax.collections[0].set_linewidth(e_sizes)
    return ax
