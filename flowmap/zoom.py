from matplotlib import pyplot as plt

from flowmapviz.zoom import plot_graph_with_zoom
from flowmapviz.plot import plot_routes
from flowmap.ax_settings import AxSettings


def get_zoom(g, segments):
    fig, ax = plt.subplots()
    plot_graph_with_zoom(g, ax, secondary_sizes=[1, 0.7, 0.5, 0.3])

    nodes_from = []
    nodes_to = []
    for seg in segments:
        nodes_from.append(seg[0])
        nodes_to.append(seg[1])

    densities = [[1]] * len(nodes_from)
    speeds = [[100]] * len(nodes_from)

    plot_routes(
        g,
        ax=ax,
        nodes_from=nodes_from,
        nodes_to=nodes_to,
        densities=densities,
        speeds=speeds,
    )
    plt.show()
    return AxSettings(ax)
