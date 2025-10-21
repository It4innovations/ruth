from datetime import datetime, timedelta

import h5py
from dask.distributed import Client, progress
import matplotlib as mpl

from .ax_settings import AxSettings
from .flowmapframe.zoom import plot_graph_with_zoom

mpl.use("Agg")
from matplotlib import pyplot as plt

from ..data.map import Map
from .flowmapframe.speeds import (plot_routes as plot_routes_speeds,
                                  get_color_bar_info as get_color_bar_info_speeds)
from .flowmapframe.plot import (plot_routes as plot_routes_densities,
                                get_color_bar_info as get_color_bar_info_densities)

def round_timedelta(td):
    seconds = td.total_seconds()
    rounded_seconds = round(seconds)
    return timedelta(seconds=rounded_seconds)

def get_stats_text(total_km_driven, total_driving_time):
    text = f"Total driving time (sum of all cars): {round_timedelta(timedelta(seconds=float(total_driving_time or 0)))}\n" \
           f"Total KMs driven: {round(total_km_driven / 1000, 2)} km\n"
    return text

def prepare_base_map(g, title=''):
    fig, ax_map = plt.subplots()
    plot_graph_with_zoom(g, ax_map, secondary_sizes=[1, 0.7, 0.5, 0.3])
    size = fig.get_size_inches()
    new_size = 20
    fig.set_size_inches(new_size, new_size * size[1] / size[0])
    ax_map.set_title(title, fontsize=40)
    
    ax_traffic = ax_map.twinx()
    ax_map_settings = AxSettings(ax_map)

    return fig, ax_map, ax_traffic, ax_map_settings

def prepare_texboxes(fig, timestamp_from, interval, total_computation_time, description=None):
    finished_vehicles_text = fig.text(
        0.7,
        0.08,
        "test",
        ha='left',
        fontsize=15)

    stats_text = fig.text(
        0.7,
        0.03,
        get_stats_text(0, 0))

    time_text = fig.text(0.15,
                            0.09,
                            datetime.utcfromtimestamp(timestamp_from * 1000 * interval // 10 ** 3),
                            ha='left',
                            fontsize=20)

    computation_time = timedelta(seconds=total_computation_time)
    fig.text(
        0.15,
        0.08,
        f"Computation time: {computation_time}\n",
        ha='left',
        va='top',
        fontsize=10)

    if description:
        txt = fig.text(
            0.15,
            0.07,
            description,
            ha='left',
            va='top',
            fontsize=10,
            wrap=True)
        txt._get_wrap_line_width = lambda: 1000

    return finished_vehicles_text, stats_text, time_text

def prepare_color_bar(fig, ax_map, plot_style_settings):
    if plot_style_settings['style'] == 'speeds':
        cmap, norm = get_color_bar_info_speeds()
        bar_title = "Speed (kph)"
    else:
        cmap, norm = get_color_bar_info_densities(0, plot_style_settings['max_width_count'])
        bar_title = "Number of vehicles"

    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar = fig.colorbar(mappable=mappable, ax=ax_map,
                        shrink=0.4, label=bar_title, pad=0.0)
    ticks = cbar.ax.get_yticks()
    cbar.ax.set_yticks(ticks)
    cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=15)
    label = cbar.ax.yaxis.label
    label.set_fontsize(15)

    if plot_style_settings['style'] == 'speeds':
        ticklabs = cbar.ax.get_yticklabels()
        ticklabs[-1] = ""
        cbar.ax.set_yticklabels(ticklabs)

    return cbar

def init_worker_graph(bbox, download_date):
    global global_g
    if global_g is None:
        global_g = Map(bbox, download_date=download_date, with_speeds=True).network
    return global_g


def plot_frames_batch(frame_ids, simulation_path, save_path,
                      bbox, download_date, timestamp_from, interval,
                      total_computation_time=0,
                      number_of_vehicles=0, title='', description=None,
                      plot_style_settings=None):
    # global global_g
    # g = global_g
    g = Map(bbox, download_date=download_date, with_speeds=True).network
    fig, ax_map, ax_traffic, ax_map_settings = prepare_base_map(g, title)
    prepare_color_bar(fig, ax_map, plot_style_settings)
    finished_vehicles_text, stats_text, time_text = prepare_texboxes(fig, timestamp_from, interval,
                                                                     total_computation_time, description)

    saved_paths = []
    with h5py.File(simulation_path, 'r') as f:
        index = f['grouped_index']
        data = f['grouped_data']

        for frame_id in frame_ids:
            # get slice for this frame
            start_offset = index[frame_id]['start_index']
            end_offset = index[frame_id]['end_index']
            records = data[start_offset:end_offset]

            active_vehicles = index[frame_id]['active_vehicles']
            vehicles_finished = index[frame_id]['vehicles_finished']
            total_distance = index[frame_id]['total_distance']
            total_driving_time = index[frame_id]['total_time']
            timestamp = index[frame_id]['rounded_ts']

            # Update text boxes -----------------------------------
            # if frame_id % 5 * 60 == 0:
            time_text.set_text(datetime.utcfromtimestamp(timestamp * 1000 * interval // 10 ** 3))

            text = f'Finished vehicles: {vehicles_finished} / {number_of_vehicles}\n' \
            f'Active vehicles: {active_vehicles}'
            finished_vehicles_text.set_text(text)
            text = get_stats_text(total_distance, total_driving_time)
            stats_text.set_text(text)

    # Plotting ---------------------------------------------
            ax_traffic.clear()
            ax_map_settings.apply(ax_traffic)
            ax_traffic.axis('off')

            nodes_from = records['node_from']
            nodes_to = records['node_to']
            vehicle_count_first = records['count_unique_first']
            vehicle_count_second = records['count_unique_second']
            vehicle_counts = [[f, s] for f, s in zip(vehicle_count_first, vehicle_count_second)]

            max_width_count = plot_style_settings['max_width_count'] if 'max_width_count' in plot_style_settings else 50
            width_modif = plot_style_settings['width_modif'] if 'width_modif' in plot_style_settings else 1

            if plot_style_settings['style'] == 'density':
                plot_routes_densities(
                    g,
                    ax=ax_traffic,
                    nodes_from=nodes_from,
                    nodes_to=nodes_to,
                    densities=vehicle_counts,
                    max_width_density=max_width_count,
                    width_modifier=width_modif,
                )

            else:
                speed_count_first = records['speed_count_first']
                speed_sum_first = records['speed_sum_first']
                speed_count_second = records['speed_count_second']
                speed_sum_second = records['speed_sum_second']
                speeds = []
                for count1, sum1, count2, sum2 in zip(speed_count_first, speed_sum_first, speed_count_second,
                                                      speed_sum_second):
                    avg1 = sum1 / count1 if count1 > 0 else -1
                    avg2 = sum2 / count2 if count2 > 0 else -1
                    speeds.append([avg1, avg2])

                plot_routes_speeds(
                    g,
                    ax=ax_traffic,
                    nodes_from=nodes_from,
                    nodes_to=nodes_to,
                    densities=vehicle_counts,
                    speeds=speeds,
                    max_width_density=max_width_count,
                    width_modifier=width_modif,
                )


            path_to_save = os.path.join(save_path, f"frame_{frame_id:04d}.png")
            fig.savefig(path_to_save, dpi=100, bbox_inches=None)
            saved_paths.append(path_to_save)

    plt.close(fig)
    return saved_paths


def setup_output_dir(path):
    os.makedirs(path, exist_ok=True)


def create_frames_dask(dask_workers,
                        simulation_path, save_path, num_of_frames,
                        bbox, download_date,
                        timestamp_from, interval,
                        total_computation_time=0,
                        number_of_vehicles=0, title='',
                        description=None,
                        plot_style_settings=None,
                        batch_size=10):
    print("Max width count:", plot_style_settings['max_width_count'])
    client = Client(threads_per_worker=2, n_workers=dask_workers)
    setup_output_dir(save_path)
    # client.run(init_worker_graph, bbox=bbox, download_date=download_date)

    # split frames into batches
    frames = list(range(num_of_frames))
    batches = [frames[i:i + batch_size] for i in range(0, len(frames), batch_size)]

    futures = [
        client.submit(
            plot_frames_batch,
            batch,
            simulation_path,
            save_path,
            bbox,
            download_date,
            timestamp_from,
            interval,
            total_computation_time,
            number_of_vehicles,
            title,
            description,
            plot_style_settings,
        ) for batch in batches
    ]

    progress(futures)
    results = client.gather(futures)
    # flatten the list of lists
    all_paths = [path for batch_paths in results for path in batch_paths]
    print(f"Saved {len(all_paths)} frames in {save_path}")
    return all_paths


# -------------- FFmpeg video saving -----------------

import os
import subprocess

def save_frames_to_video(frames_dir, output_path, fps):
    input_pattern = os.path.join(frames_dir, "frame_%04d.png")

    # Try MP4 first
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-i", input_pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_path
    ]

    try:
        subprocess.run(cmd, check=True,  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Saved video: {output_path}")
    except Exception:
        # fallback to GIF
        output_gif = output_path.replace(".mp4", ".gif")
        cmd_gif = [
            "ffmpeg",
            "-y",
            "-framerate", str(fps),
            "-i", input_pattern,
            "-vf", f"fps={fps},scale=640:-1:flags=lanczos",
            output_gif
        ]
        subprocess.run(cmd_gif, check=True,  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"FFmpeg not available or failed, saved GIF instead: {output_gif}")
