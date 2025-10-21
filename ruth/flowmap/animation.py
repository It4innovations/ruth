from datetime import datetime, timedelta
from time import time
from os import path
from abc import ABC, abstractmethod

import logging
import h5py
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt, animation
from shapely import LineString
from tqdm import tqdm

from .flowmapframe.zoom import plot_graph_with_zoom
from .flowmapframe.plot import get_node_coordinates, WidthStyle
from .flowmapframe.speeds import (plot_routes as plot_routes_speeds,
                                  get_color_bar_info as get_color_bar_info_speeds)
from .flowmapframe.plot import (plot_routes as plot_routes_densities,
                                get_color_bar_info as get_color_bar_info_densities)

from ruth.utils import TimerSet
from ruth.simulator import Simulation

from .ax_settings import AxSettings
from .plot_dask import create_frames_dask, save_frames_to_video
from ..data.map import Map, BBox

def load_file_content(path):
    with open(path, 'r') as f:
        return f.read()


def round_timedelta(td):
    seconds = td.total_seconds()
    rounded_seconds = round(seconds)
    return timedelta(seconds=rounded_seconds)


def load_description(path):
    description = load_file_content(path)
    if len(description) > 990:
        print('Warning: description is too long. The end might not be visible in the animation.')
    return description


class SimulationAnimator(ABC):
    def __init__(self, simulation_path, fps, save_path, frame_start, frames_len, width_modif, title, description,
                 description_path, max_width_count, gif, dask_workers):
        self.simulation_path = simulation_path
        self.simulation = None
        self.fps = fps
        self.save_path = save_path
        self.frame_start = frame_start
        self.frames_len = frames_len
        self.width_modif = width_modif
        self.title = title
        self.speed = None
        self.max_width_count = max_width_count
        self.generate_gif = gif
        self.dask = dask_workers > 0
        self.dask_workers = dask_workers

        self.description = None
        if description:
            self.description = description
        elif description_path:
            self.description = load_description(description_path)

        self.ts = TimerSet()

        # added during preprocessing
        self.bbox = None
        self.map_download_date = None

        self.timestamp_from = None
        self.num_of_frames = None
        self.computation_by_simulation_time = None
        self.timed_seg_dict = None
        self.number_of_vehicles = None


    @property
    @abstractmethod
    def colorbar_title(self):
        pass

    def plot_cbar(self):
        cmap, norm = self.get_color_bar_info()
        cbar = plt.colorbar(mappable=mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=self.ax_map,
                            shrink=0.4, label=self.colorbar_title(), pad=0.0)
        ticks = cbar.ax.get_yticks()
        cbar.ax.set_yticks(ticks)
        cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=15)
        label = cbar.ax.yaxis.label
        label.set_fontsize(15)
        return cbar

    def preprocess(self, simulation=None):
        with self.ts.get("data loading"):
            with h5py.File(self.simulation_path, 'r') as f:
                bbox = f.attrs['bbox']
                self.bbox = BBox(*bbox)
                self.map_download_date = f.attrs['download_date']
                self.timestamp_from = f['grouped_index'][0][0]
                self.total_computation_time = f.attrs.get('computational_time', None)
                self.num_of_frames = len(f['grouped_index'])
                self.interval = int(f.attrs['interval_s']) if 'interval_s' in f.attrs else 5
                self.number_of_vehicles = int(f.attrs['number_of_vehicles']) if 'number_of_vehicles' in f.attrs else '-'

                if self.frames_len is not None:
                    self.num_of_frames = min(self.num_of_frames - self.frame_start, self.frames_len)

                calculated_max_width_count = int(f.attrs['max_unique_vehicles_on_segment']) if 'max_unique_vehicles_on_segment' in f.attrs else 1
                if self.max_width_count is None:
                    self.max_width_count = calculated_max_width_count

                print(f"Loaded data from {self.simulation_path}:")
                print(f" - BBox: {self.bbox}")
                print(f" - Map download date: {self.map_download_date}")
                print(f" - Simulation start time: {datetime.utcfromtimestamp(self.timestamp_from * self.interval)}")
                print(f" - Number of frames: {self.num_of_frames}")
                print(f" - Interval between frames: {self.interval} seconds")
                print(f" - Total number of vehicles: {self.number_of_vehicles}")
                if self.total_computation_time is not None:
                    print(f" - Total computation time: {timedelta(seconds=self.total_computation_time)}")

    def run(self, simulation=None):
        self.preprocess(simulation)

        # self._set_ax_settings_if_zoom()
        self.g = Map(self.bbox, download_date=self.map_download_date, with_speeds=True).network

        with self.ts.get("base map preparing"):
            if not self.dask:
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

    def _load_data(self, simulation):
        # pickle load
        if simulation is None and self.simulation_path.endswith('.pickle'):
            simulation = Simulation.load(self.simulation_path)

        # simulation object
        start_time = time()
        if simulation is not None and isinstance(simulation, Simulation):
            df = simulation.history.to_dataframe_short()
            not_finished_vehicles = simulation.get_vehicle_ids_not_finished()

            si_df = simulation.steps_info_to_dataframe()
            departure_time = simulation.setting.departure_time
            self.bbox = simulation.bbox
            self.map_download_date = simulation.map_download_date
            self.total_computation_time = simulation.duration.total_seconds()
        # hdf5
        elif self.simulation_path.endswith(('.hdf5', '.h5')):
            result = Simulation.load_h5_df(self.simulation_path)
            df = result['df']
            departure_time = result['departure_time']
            self.bbox = result['bbox']
            self.map_download_date = result['download_date']
            self.total_computation_time = result.get('computational_time', None)
            not_finished_vehicles = set(df.groupby('vehicle_id')['active'].all()[lambda x: x].index)
            df.drop(columns=['active'], inplace=True)

            si_df = None
        else:
            raise NotImplementedError
        logging.info(f"Data loaded in {round(time() - start_time, 5)} s")
        return df, si_df, departure_time, not_finished_vehicles

    def _preprocess_data(self):
        raise NotImplementedError("This method was deprecated. Use preprocess method instead.")

    def _set_ax_settings_if_zoom(self):
        raise NotImplementedError("Zooming is not implemented in the base class. Use a subclass that defines self.segments.")
        #  if self.zoom:
            # self.segments = all segments with data in the time range
            # print('Use the zoom button to choose an area that will be zoomed in in the animation.')
            # print('Close the window when satisfied with the selection.')
            # self.ax_map_settings = get_zoom(self.g, self.segments)

    def _get_stats_text(self, total_km_driven, total_driving_time):
        text = f"Total driving time (sum of all cars): {round_timedelta(timedelta(seconds=float(total_driving_time or 0)))}\n" \
               f"Total KMs driven: {round(total_km_driven / 1000, 2)} km\n"
        return text

    def _get_finished_vehicles_text(self, active_vehicles: int, vehicles_finished: int):
        return \
            f'Finished vehicles: {vehicles_finished} / {self.number_of_vehicles}\n' \
            f'Active vehicles: {active_vehicles}'

    def _prepare_base_map(self):
        mpl.use('Agg')

        self.fig, self.ax_map = plt.subplots()
        plot_graph_with_zoom(self.g, self.ax_map, secondary_sizes=[1, 0.7, 0.5, 0.3])

        # if self.zoom:
        #     self.ax_map_settings.apply(self.ax_map)

        size = self.fig.get_size_inches()
        new_size = 20
        size[1] = size[1] * new_size / size[0]
        size[0] = new_size
        self.fig.set_size_inches(size)

        plt.title(self.title, fontsize=40)
        self.time_text = plt.figtext(
            0.15,
            0.09,
            datetime.utcfromtimestamp(self.timestamp_from * 1000 * self.interval // 10 ** 3),
            ha='left',
            fontsize=20)

        self.finished_vehicles_text = plt.figtext(
            0.7,
            0.08,
            "",
            ha='left',
            fontsize=15)

        if self.description:
            txt = plt.figtext(
                0.15,
                0.07,
                self.description,
                ha='left',
                va='top',
                fontsize=10,
                wrap=True)
            txt._get_wrap_line_width = lambda: 1000

        if self.computation_by_simulation_time is not None:
            computation_time = timedelta(milliseconds=self.computation_by_simulation_time[self.timestamp_from])
            self.compute_text = plt.figtext(
                0.15,
                0.08,
                f"Computation time: {computation_time}\n",
                ha='left',
                va='top',
                fontsize=10)
        elif self.total_computation_time is not None:
            computation_time = timedelta(seconds=self.total_computation_time)
            self.compute_text = plt.figtext(
                0.15,
                0.08,
                f"Computation time: {computation_time}\n",
                ha='left',
                va='top',
                fontsize=10)

        self.stats_text = plt.figtext(
            0.7,
            0.03,
            self._get_stats_text(0, 0))

        self.plot_cbar()

        self.ax_traffic = self.ax_map.twinx()
        self.ax_map_settings = AxSettings(self.ax_map)

    def _create_animation(self):
        timestamp = round(time() * 1000)
        filetype = 'gif' if self.generate_gif else 'mp4'
        output_path = path.join(self.save_path, f'{str(timestamp)}-rt.{filetype}')

        if self.dask:
            plot_style_settings = {}
            plot_style_settings['style'] = 'speeds' if isinstance(self, SimulationSpeedsAnimator) else 'density'
            plot_style_settings['width_modif'] = self.width_modif
            plot_style_settings['max_width_count'] = self.max_width_count
            plot_style_settings['width_style'] = self.width_style if isinstance(self, SimulationVolumeAnimator) else None

            # name frames folder with timestamp to avoid conflicts
            frames_folder_path = path.join(self.save_path, f'frames-{str(timestamp)}')
            create_frames_dask(self.dask_workers,
                                self.simulation_path,
                                frames_folder_path,
                                self.num_of_frames,
                                self.bbox,
                                self.map_download_date,
                                self.timestamp_from,
                                self.interval,
                                self.total_computation_time,
                                self.number_of_vehicles,
                                self.title,
                                self.description,
                                plot_style_settings)
            save_frames_to_video(frames_folder_path, output_path, fps=self.fps)
        else:
            anim = animation.FuncAnimation(
                plt.gcf(),
                self._animate(),
                interval=75,
                frames=self.num_of_frames,
                repeat=False
            )
            anim.save(output_path, writer='ffmpeg', fps=self.fps)

    def _animate(self):
        progress_bar = None

        def step(i):
            nonlocal progress_bar
            with h5py.File(self.simulation_path, 'r') as f:

                data = f['grouped_data']
                index = f['grouped_index']
                if progress_bar is None:
                    progress_bar = tqdm(total=self.num_of_frames, desc='Creating animation', unit='frame', leave=True)

                timestamp = index[i]['rounded_ts']

                start_offset = index[i]['start_index']
                end_offset = index[i]['end_index']
                records = data[start_offset:end_offset]

                active_vehicles = index[i]['active_vehicles']
                vehicles_finished = index[i]['vehicles_finished']
                total_distance = index[i]['total_distance']
                total_driving_time = index[i]['total_time']

                self.ax_traffic.clear()
                self.ax_map_settings.apply(self.ax_traffic)
                self.ax_traffic.axis('off')

                if i % 5 * 60 == 0:
                    self.time_text.set_text(datetime.utcfromtimestamp(timestamp * 1000 * self.interval // 10 ** 3))

                self.finished_vehicles_text.set_text(self._get_finished_vehicles_text(active_vehicles, vehicles_finished))
                self.stats_text.set_text(self._get_stats_text(total_distance, total_driving_time))

                if self.computation_by_simulation_time is not None:
                    computation_time = timedelta(milliseconds=self.computation_by_simulation_time[timestamp])
                    self.compute_text.set_text(f"Computation time: {computation_time}\n")

                self._plot_routes(records)

                progress_bar.update(1)
                if i == self.num_of_frames - 1:
                    progress_bar.close()

        return step

    @abstractmethod
    def _plot_routes(self, records: np.array):
        pass

    @abstractmethod
    def get_color_bar_info(self):
        pass

    def get_cars_xy(self, node_from, node_to, offsets):
        x, y = get_node_coordinates(self.g, node_from, node_to)

        line = LineString(zip(x, y))
        edge_length = self.g[node_from][node_to][0]['length']

        offsets = np.interp(offsets, [0, edge_length], [0, line.length])
        point = line.interpolate(offsets)
        return point.x, point.y


class SimulationVolumeAnimator(SimulationAnimator):
    def __init__(self, simulation_path, fps, save_path, frame_start, frames_len, width_modif, title, description,
                 description_path, max_width_count, gif, dask_workers, width_style="BOXED"):
        super().__init__(
            simulation_path,
            fps,
            save_path,
            frame_start,
            frames_len,
            width_modif,
            title,
            description,
            description_path,
            max_width_count,
            dask_workers,
            gif
        )
        self.width_style = WidthStyle[width_style]

    def _plot_routes(self, records):
        nodes_from = records['node_from']
        nodes_to = records['node_to']
        vehicle_count_first = records['count_unique_first']
        vehicle_count_second = records['count_unique_second']
        vehicle_counts = [[f, s] for f, s in zip(vehicle_count_first, vehicle_count_second)]

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
        return records

    def get_color_bar_info(self):
        return get_color_bar_info_densities(0, self.max_width_count)

    def colorbar_title(self):
        return "Number of vehicles"


class SimulationSpeedsAnimator(SimulationAnimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _plot_routes(self, records):
        nodes_from = records['node_from']
        nodes_to = records['node_to']
        vehicle_count_first = records['count_unique_first']
        vehicle_count_second = records['count_unique_second']
        vehicle_counts = [[f, s] for f, s in zip(vehicle_count_first, vehicle_count_second)]
        speed_count_first = records['speed_count_first']
        speed_sum_first = records['speed_sum_first']
        speed_count_second = records['speed_count_second']
        speed_sum_second = records['speed_sum_second']
        speeds = []
        for count1, sum1, count2, sum2 in zip(speed_count_first, speed_sum_first, speed_count_second, speed_sum_second):
            avg1 = sum1 / count1 if count1 > 0 else -1
            avg2 = sum2 / count2 if count2 > 0 else -1
            speeds.append([avg1, avg2])

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
        return

    def get_color_bar_info(self):
        return get_color_bar_info_speeds()

    def colorbar_title(self):
        return "Speed (kph)"

    def plot_cbar(self):
        cbar = super().plot_cbar()
        ticklabs = cbar.ax.get_yticklabels()
        ticklabs[-1] = ""
        cbar.ax.set_yticklabels(ticklabs)
        return cbar
