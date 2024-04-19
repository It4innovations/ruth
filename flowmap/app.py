import os
import click
import pathlib
import platform
from ruth.simulator import Simulation
from flowmap.flowmapframe.plot import WidthStyle
from flowmap.analysis import create_simulations_comparison
from flowmap.time_unit import TimeUnit
from flowmap.animation import SimulationVolumeAnimator, SimulationSpeedsAnimator
from flowmap.info import SimulationInfo, get_real_time


def set_path():
    # simulations generated on Linux can't be loaded on Windows and vice versa without changing the path
    if platform.system() == "Windows":
        pathlib.PosixPath = pathlib.WindowsPath
    elif platform.system() == "Linux":
        pathlib.WindowsPath = pathlib.PosixPath


@click.group()
def cli():
    pass


animation_options = {
    'fps': {'default': 25, 'type': int, 'help': "Set video frames per second.", 'show_default': True},
    'save_path': {'default': '', 'help': "Path to the folder for the output video."},
    'frame_start': {'default': 0, 'type': int, 'help': "Number of frames to skip before plotting."},
    'frames_len': {'type': int, 'help': "Number of frames to plot"},
    'width_modif': {'default': 10, 'type': click.IntRange(2, 200, clamp=True), 'show_default': True,
                    'help': "Adjust width."},
    'title': {'default': '', 'help': "Set video title"},
    'description_path': {'default': '', 'help': "Path to the file with description to be added to the video."},
    'speed': {'type': int, 'help': "Set video speed."},
    'divide': {'default': 2, 'type': int, 'help': "Into how many parts will each segment be split.", 'show_default': True},
    'max_width_count': {'default': None, 'type': int, 'help': "Number of vehicles that corresponds to the maximum width of the segment. If not specified, it will be set dynamically according to the data.", 'show_default': True},
    'plot_cars': {'is_flag': True, 'help': "Visualize cars on the map."},
    'zoom': {'is_flag': True, 'help': "Choose zoom manually."},
}


def click_animation_options(function):
    for option, value in animation_options.items():
        function = click.option(f'--{option}', **value)(function)
    return function


def click_animation_common(function):
    function = click.argument('simulation_path')(function)
    function = click_animation_options(function)
    return function


@cli.command()
@click_animation_common
@click.option(
    '--width-style',
    type=click.Choice([el.name for el in WidthStyle]),
    default='BOXED',
    help="Choose style of width plotting"
)
def generate_volume_animation(**kwargs):
    simulation_animator = SimulationVolumeAnimator(**kwargs)
    simulation_animator.run()


@cli.command()
@click_animation_common
def generate_speeds_animation(**kwargs):
    simulation_animator = SimulationSpeedsAnimator(**kwargs)
    simulation_animator.run()


@cli.command()
@click.argument('simulation-path', type=click.Path(exists=True))
@click.option(
    '--time-unit',
    type=str,
    help="Time unit. Possible values: [seconds|minutes|hours]",
    default='hours'
)
@click.option('--speed', default=1, help="Speed up the video.", show_default=True)
@click.option(
    '--minute',
    type=int,
    default=None,
    help="Set if you want to get more detailed info about nth minute of the simulation. Takes longer to calculate",
    show_default=True
)
@click.option(
    '--status-at-point',
    type=float,
    default=None,
    help="Set if you want to get info about the status of the simulation at a given state of completion (0-1). Takes "
         "longer to calculate",
    show_default=True
)
def get_info(simulation_path, time_unit, speed, minute, status_at_point):
    sim = Simulation.load(simulation_path)

    time_unit = TimeUnit.from_str(time_unit)
    time_unit_minutes = TimeUnit.from_str('minutes')

    real_time = get_real_time(sim, time_unit)
    print(f'Real time duration: {real_time} {time_unit.name.lower()}.')

    real_time_minutes = get_real_time(sim, time_unit_minutes)

    print(f'Video length: {real_time_minutes / speed} minutes.')

    simulation_info = None

    if minute is not None:
        simulation_info = SimulationInfo(sim)
        simulation_info.print_info(minute)

    if status_at_point is not None:
        if simulation_info is None:
            simulation_info = SimulationInfo(sim)
        simulation_info.print_status_at_point(status_at_point)


@cli.command()
@click.argument('input-dir', type=click.Path(exists=True))
@click.option('--output-dir', '-o', default='', help="Path to the folder for the output files.")
@click.option(
    '--interval',
    type=int,
    default=5,
    help="Time interval in minutes",
    show_default=True
)
def get_comparison_csv(input_dir, output_dir, interval):
    pickles = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".pickle"):
            pickles.append(os.path.join(input_dir, filename))
        else:
            continue
    create_simulations_comparison(pickles, output_dir, interval)


def main():
    cli()


if __name__ == '__main__':
    set_path()
    main()
