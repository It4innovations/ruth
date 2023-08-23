import click

from ruth.simulator import Simulation
from flowmap.flowmapframe.plot import WidthStyle

from flowmap.time_unit import TimeUnit
from flowmap.animation import SimulationVolumeAnimator, SimulationSpeedsAnimator
from flowmap.info import SimulationInfo, get_real_time


@click.group()
def cli():
    pass


def click_animation_common(function):
    function = click.argument('simulation_path')(function)
    function = click.option('--fps', '-f', default=25, help="Set video frames per second.", show_default=True)(function)
    function = click.option('--save-path', default='', help="Path to the folder for the output video.")(function)
    function = click.option('--frame-start', default=0, help="Number of frames to skip before plotting.")(function)
    function = click.option('--frames-len', help="Number of frames to plot")(function)
    function = click.option(
        '--width-modif',
        default=10,
        type=click.IntRange(2, 200, clamp=True),
        show_default=True,
        help="Adjust width."
    )(function)
    function = click.option('--title', '-t', default='', help="Set video title")(function)
    function = click.option('--speed', '-s', default=1, help="Set video speed.")(function)
    function = click.option(
        '--divide',
        '-d',
        default=2,
        help="Into how many parts will each segment be split.",
        show_default=True
    )(function)
    function = click.option(
        '--max-width-count',
        '-m',
        default=None,
        help="Number of vehicles that corresponds to the maximum width of the segment. If not specified, it will be set "
                "dynamically according to the data.",
        show_default=True
    )(function)
    function = click.option('--plot-cars', '-c', is_flag=True, help="Visualize cars on the map.")(function)
    function = click.option('--zoom', '-z', is_flag=True, help="Choose zoom manually.")(function)
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


def main():
    cli()


if __name__ == '__main__':
    main()
