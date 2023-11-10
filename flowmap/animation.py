from datetime import datetime, timedelta
from time import time
from os import path
from abc import ABC, abstractmethod

import logging
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt, animation
from shapely import LineString
from tqdm import tqdm
from collections import defaultdict

from flowmap.flowmapframe.zoom import plot_graph_with_zoom
from flowmap.flowmapframe.plot import get_node_coordinates, WidthStyle
from flowmap.flowmapframe.speeds import plot_routes as plot_routes_speeds
from flowmap.flowmapframe.plot import plot_routes as plot_routes_densities

from ruth.utils import TimerSet
from ruth.simulator import Simulation

from flowmap.ax_settings import AxSettings
from flowmap.zoom import get_zoom
from flowmap.input import preprocess_data


def load_file_content(path):
    with open(path, 'r') as f:
        return f.read()


def round_timedelta(td):
    seconds = td.total_seconds()
    rounded_seconds = round(seconds)
    return timedelta(seconds=rounded_seconds)


class SimulationAnimator(ABC):
    def __init__(self, simulation_path, fps, save_path, frame_start, frames_len,
                 width_modif, title, description_path, speed, divide, max_width_count, plot_cars, zoom):
        self.simulation_path = simulation_path
        self.fps = fps
        self.save_path = save_path
        self.frame_start = frame_start
        self.frames_len = frames_len
        self.width_modif = width_modif
        self.title = title
        self.description = load_file_content(description_path) if description_path else None
        self.speed = speed
        self.divide = divide
        self.max_width_count = max_width_count
        self.plot_cars = plot_cars
        self.zoom = zoom
        self.ts = TimerSet()

    @property
    def interval(self):
        return self.speed / self.fps

    def run(self):
        with self.ts.get("data loading"):
            logging.info('Loading simulation data...')
            self._load_data()
            logging.info('Simulation data loaded.')

        with self.ts.get("data preprocessing"):
            logging.info('Preprocessing data...')
            self._preprocess_data()
            logging.info('Data preprocessed.')

        self._set_ax_settings_if_zoom()

        with self.ts.get("base map preparing"):
            logging.info('Preparing base map...')
            self._prepare_base_map()
            logging.info('Base map prepared.')

        with self.ts.get("create animation"):
            logging.info('Creating animation...')
            self._create_animation()
            logging.info('Animation created.')

        print()
        for k, v in self.ts.collect().items():
            print(f'{k}: {v} ms')

    def _load_data(self):
        sim = Simulation.load(self.simulation_path)
        self.g = sim.routing_map.network
        self.sim_history = sim.history.to_dataframe()

    def _preprocess_data(self):
        preprocessed_data = preprocess_data(self.sim_history, self.g, self.speed, self.fps, self.divide)
        self.segments = preprocessed_data.segments
        self.number_of_vehicles = preprocessed_data.number_of_vehicles
        self.number_of_finished_vehicles_in_time = preprocessed_data.number_of_finished_vehicles_in_time
        self.total_meters_driven_in_time = preprocessed_data.total_meters_driven_in_time
        self.total_simulation_time_in_time_s = preprocessed_data.total_simulation_time_in_time_s
        timestamps = [seg.timestamp for seg in preprocessed_data.timed_segments]
        min_timestamp = min(timestamps)
        max_timestamp = max(timestamps)

        if self.max_width_count is None:
            self.max_width_count = max([max(seg.counts) for seg in preprocessed_data.timed_segments])
        self.timestamp_from = min_timestamp + self.frame_start
        self.num_of_frames = max_timestamp - self.timestamp_from + 1  # + 1 to include the last frame
        self.num_of_frames = min(int(self.frames_len), self.num_of_frames) if self.frames_len else self.num_of_frames

        self.timed_seg_dict = defaultdict(list)
        for seg in preprocessed_data.timed_segments:
            self.timed_seg_dict[seg.timestamp].append(seg)

    def _set_ax_settings_if_zoom(self):
        if self.zoom:
            print('Use the zoom button to choose an area that will be zoomed in in the animation.')
            print('Close the window when satisfied with the selection.')
            self.ax_map_settings = get_zoom(self.g, self.segments)

    def _get_stats_text(self, timestamp: int):
        simulation_time_formatted = round_timedelta(self.total_simulation_time_in_time_s[timestamp])
        total_km_driven = round(self.total_meters_driven_in_time[timestamp] / 1000, 2)
        return f"total simulation time: {simulation_time_formatted}\n" \
               f"total meters driven: {total_km_driven} km\n"

    def _get_finished_vehicles_text(self, timestamp: int):
        return f'Finished vehicles: {self.number_of_finished_vehicles_in_time[timestamp]} / {self.number_of_vehicles}'

    def _prepare_base_map(self):
        mpl.use('Agg')

        self.fig, self.ax_map = plt.subplots()
        plot_graph_with_zoom(self.g, self.ax_map, secondary_sizes=[1, 0.7, 0.5, 0.3])

        if self.zoom:
            self.ax_map_settings.apply(self.ax_map)

        size = self.fig.get_size_inches()
        new_size = 20
        size[1] = size[1] * new_size / size[0]
        size[0] = new_size
        self.fig.set_size_inches(size)

        plt.title(self.title, fontsize=40)
        self.time_text = plt.figtext(
            0.3,
            0.09,
            datetime.utcfromtimestamp(self.timestamp_from * 1000 * self.interval // 10 ** 3),
            ha='right',
            fontsize=20)

        self.finished_vehicles_text = plt.figtext(
            0.7,
            0.09,
            self._get_finished_vehicles_text(self.timestamp_from),
            ha='left',
            fontsize=20)

        if self.description:
            txt = plt.figtext(
                0.15,
                0.08,
                self.description,
                ha='left',
                va='top',
                fontsize=10,
                wrap=True)
            txt._get_wrap_line_width = lambda: 1000

        self.stats_text = plt.figtext(
            0.7,
            0.05,
            self._get_stats_text(self.timestamp_from))

        self.ax_traffic = self.ax_map.twinx()
        self.ax_map_settings = AxSettings(self.ax_map)

    def _create_animation(self):
        anim = animation.FuncAnimation(
            plt.gcf(),
            self._animate(),
            interval=75,
            frames=self.num_of_frames,
            repeat=False
        )

        timestamp = round(time() * 1000)
        anim.save(path.join(self.save_path, str(timestamp) + '-rt.mp4'), writer='ffmpeg', fps=self.fps)

    def _animate(self):
        car_coordinates = []
        progress_bar = None

        def step(i):
            nonlocal car_coordinates, progress_bar
            if progress_bar is None:
                progress_bar = tqdm(total=self.num_of_frames, desc='Creating animation', unit='frame', leave=True)
            self.ax_traffic.clear()
            self.ax_map_settings.apply(self.ax_traffic)
            self.ax_traffic.axis('off')

            timestamp = self.timestamp_from + i
            if i % 5 * 60 == 0:
                self.time_text.set_text(datetime.utcfromtimestamp(timestamp * 1000 * self.interval // 10 ** 3))

            self.finished_vehicles_text.set_text(self._get_finished_vehicles_text(timestamp))
            self.stats_text.set_text(self._get_stats_text(timestamp))

            segments = self._plot_routes(timestamp)

            if self.plot_cars:
                if len(car_coordinates) == 3:
                    car_coordinates.pop(0)
                xp = []
                yp = []
                for segment in segments:
                    if segment.cars_offsets is not None:
                        x, y = self.get_cars_xy(segment.node_from.id, segment.node_to.id, segment.cars_offsets)
                        xp.append(x)
                        yp.append(y)
                alphas = [1, 0.75, 0.5]
                car_coordinates.append((xp, yp))
                for coords, alpha in zip(reversed(car_coordinates), alphas):
                    self.ax_traffic.scatter(coords[0], coords[1], facecolors='none', edgecolors='black', alpha=alpha)

            progress_bar.update(1)
            if i == self.num_of_frames - 1:
                progress_bar.close()

        return step

    @abstractmethod
    def _plot_routes(self, timestamp):
        pass

    def get_cars_xy(self, node_from, node_to, offsets):
        x, y = get_node_coordinates(self.g, node_from, node_to)

        line = LineString(zip(x, y))
        edge_length = self.g[node_from][node_to][0]['length']

        offsets = np.interp(offsets, [0, edge_length], [0, line.length])
        point = line.interpolate(offsets)
        return point.x, point.y


class SimulationVolumeAnimator(SimulationAnimator):
    def __init__(self, simulation_path, fps, save_path, frame_start, frames_len, width_style,
                 width_modif, title, speed, divide, max_width_count, plot_cars, zoom):
        super().__init__(
            simulation_path,
            fps,
            save_path,
            frame_start,
            frames_len,
            width_modif,
            title,
            speed,
            divide,
            max_width_count,
            plot_cars,
            zoom
        )
        self.width_style = WidthStyle[width_style]

    def _plot_routes(self, timestamp):
        segments = self.timed_seg_dict[timestamp]
        nodes_from = [seg.node_from.id for seg in segments]
        nodes_to = [seg.node_to.id for seg in segments]
        vehicle_counts = [seg.counts for seg in segments]

        plot_routes_densities(
            self.g,
            ax=self.ax_traffic,
            nodes_from=nodes_from,
            nodes_to=nodes_to,
            densities=vehicle_counts,
            max_width_density=self.max_width_count,
            width_modifier=self.width_modif,
            width_style=self.width_style
        )
        return segments


class SimulationSpeedsAnimator(SimulationAnimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _plot_routes(self, timestamp):
        segments = self.timed_seg_dict[timestamp]
        nodes_from = [seg.node_from.id for seg in segments]
        nodes_to = [seg.node_to.id for seg in segments]
        vehicle_counts = [seg.counts for seg in segments]
        speeds = [seg.speeds for seg in segments]

        plot_routes_speeds(
            self.g,
            ax=self.ax_traffic,
            nodes_from=nodes_from,
            nodes_to=nodes_to,
            densities=vehicle_counts,
            speeds=speeds,
            max_width_density=self.max_width_count,
            width_modifier=self.width_modif
        )
        return segments
