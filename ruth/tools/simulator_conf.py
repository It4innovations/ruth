import logging
import os
from dataclasses import dataclass, make_dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import click
from click import IntRange
from serde import serde, field, Strict
from serde.json import from_json

from ..flowmap import animation
from ..flowmap.app import animation_options
from ..simulator import Simulation
from ..tools.simulator import (run_inner, AlternativesRatio as AlternativesRatioInner, CommonArgs as CommonArgsInner,
                               RouteSelectionRatio as RouteSelectionRatioInner, animate)


def make_animation_args_dataclass(options_dict):
    fields = []
    for key, value in options_dict.items():
        field_type = value.get('type', None)
        default_value = value.get('default', None)
        if 'is_flag' in value:
            default_value = False
            field_type = bool

        if isinstance(field_type, IntRange):
            field_type = int

        fields.append((key, field_type, default_value))

    return make_dataclass("AnimationArgs", fields)


@serde(rename_all="kebabcase", type_check=Strict)
@dataclass
class CommonArgs(CommonArgsInner):
    task_id: Optional[str] = None
    departure_time: datetime = field(serializer=lambda x: x.strftime("%Y-%m-%d %H:%M:%S"),
                                     deserializer=lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"),
                                     default=datetime.now())
    round_frequency: timedelta = field(rename="round-frequency-s",
                                       serializer=lambda x: x.total_seconds(),
                                       deserializer=lambda x: timedelta(seconds=x),
                                       default=timedelta(seconds=5))
    k_alternatives: int = 1
    map_update_freq: timedelta = field(rename="map-update-freq-s",
                                       serializer=lambda x: x.total_seconds(),
                                       deserializer=lambda x: timedelta(seconds=x),
                                       default=timedelta(seconds=1))
    los_vehicles_tolerance: timedelta = field(rename="los-vehicles-tolerance-s",
                                              serializer=lambda x: x.total_seconds(),
                                              deserializer=lambda x: timedelta(seconds=x),
                                              default=timedelta(seconds=5))
    travel_time_limit_perc: float = field(rename="travel-time-limit-perc",
                                          default=0.0)
    speeds_path: Optional[str] = field(serializer=lambda x: '' if x is None else x,
                                       deserializer=lambda x: None if x == '' else x,
                                       default=None)
    out: str = "simulation-record.pickle"
    seed: Optional[int] = None
    walltime: Optional[timedelta] = field(rename="walltime-s",
                                          serializer=lambda x: x.total_seconds(),
                                          deserializer=lambda x: timedelta(seconds=x),
                                          default=None)
    saving_interval: Optional[timedelta] = field(rename="saving-interval-s",
                                                 serializer=lambda x: x.total_seconds(),
                                                 deserializer=lambda x: timedelta(seconds=x),
                                                 default=None)
    continue_from: Optional[Simulation] = field(serializer=lambda x: x.store("continue-from.pickle"),
                                                deserializer=lambda x: None if x == "" else Simulation.load(x),
                                                default=None)
    stuck_detection: int = 0
    plateau_default_route: bool = False


@serde(rename_all="kebabcase")
@dataclass
class RunArgs:
    vehicles_path: Optional[str] = None


@serde(rename_all="kebabcase")
class DistributedArgs:
    number_of_workers: int
    evkit_dir_path: str = "evkit"
    spawn_workers_at_main_node: bool = True
    try_to_kill: bool = False


@serde(rename_all="kebabcase")
@dataclass
class AlternativesRatio(AlternativesRatioInner):
    default: float = 0.0
    dijkstra_fastest: float = 0.0
    dijkstra_shortest: float = 0.0
    plateau_fastest: float = 0.0


@serde(rename_all="kebabcase")
@dataclass
class RouteSelectionRatio(RouteSelectionRatioInner):
    no_alternative: float = 0.0
    first: float = 0.0
    random: float = 0.0
    ptdr: float = 0.0


AnimationArgs = make_animation_args_dataclass(animation_options)


@serde(rename_all="kebabcase")
@dataclass
class Args:
    common: CommonArgs = field(rename="ruth-simulator")
    run: RunArgs = field(rename="run")
    alternatives_ratio: AlternativesRatio = field(rename="alternatives")
    route_selection_ratio: RouteSelectionRatio = field(rename="route-selection")
    distribution: DistributedArgs = field(rename="distribution", default=None)
    animation: AnimationArgs = field(rename="animation", default=None)


def fill_args(config_file, ctx=None, debug=False):
    if os.path.isfile(config_file):
        logging.info(f"Settings taken from config file {config_file}.")
        with open(config_file, 'r') as f:
            config_data = f.read()
            args = from_json(Args, config_data)
    else:
        logging.info(f"Config file not found.")
        args = Args(CommonArgs(), RunArgs(), AlternativesRatio(), RouteSelectionRatio(), AnimationArgs())

    p = Path(args.run.vehicles_path) if args.run.vehicles_path is not None else None

    if ctx is not None:
        ctx.obj['DEBUG'] = debug
        ctx.obj['common-args'] = args.common
        ctx.obj['run-args'] = args.run
        ctx.obj['alternatives-ratio'] = args.alternatives_ratio
        ctx.obj['route-selection-ratio'] = args.route_selection_ratio
        ctx.obj['animation'] = args.animation
        ctx.obj['path'] = p

    return args, p


@click.group(chain=True)
@click.option('--config-file', default='config.json')
@click.option('--debug/--no-debug', default=False)
@click.pass_context
def single_node_simulator_conf(ctx,
                               config_file,
                               debug):
    # ensure that ctx.obj exists and is a dict (in case `cli()` is called by means other than the `if` block bellow)
    ctx.ensure_object(dict)
    fill_args(config_file, ctx, debug)


@single_node_simulator_conf.command()
@click.pass_context
def run(ctx):
    common_args = ctx.obj["common-args"]
    run_args = ctx.obj["run-args"]
    alternatives_ratio = ctx.obj["alternatives-ratio"]
    route_selection_ratio = ctx.obj["route-selection-ratio"]
    p = ctx.obj["path"]
    ctx.obj['simulation'] = run_inner(common_args, p, alternatives_ratio, route_selection_ratio)


@single_node_simulator_conf.command()
@click.pass_context
def volume_animation(ctx):
    animate(ctx, animation.SimulationVolumeAnimator, **ctx.obj["animation"].__dict__)


@single_node_simulator_conf.command()
@click.pass_context
def speed_animation(ctx):
    animate(ctx, animation.SimulationSpeedsAnimator, **ctx.obj["animation"].__dict__)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s:%(levelname)-4s %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
        force=True
    )
    single_node_simulator_conf(obj={})


if __name__ == "__main__":
    main()
