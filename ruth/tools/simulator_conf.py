import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import click
from serde import serde, field, Strict
from serde.json import from_json

from ..simulator import Simulation
from .. tools.simulator import (run_inner, AlternativesRatio as AlternativesRatioInner, CommonArgs as CommonArgsInner,
                                RouteSelectionRatio as RouteSelectionRatioInner)


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
    disable_stuck_detection: bool = False
    plateau_default_route: bool = False



@serde(rename_all="kebabcase")
@dataclass
class RunArgs:
    vehicles_path: Optional[str] = None


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


@serde(rename_all="kebabcase")
@dataclass
class Args:
    common: CommonArgs = field(rename="ruth-simulator")
    run: RunArgs = field(rename="run")
    alternatives_ratio: AlternativesRatio = field(rename="alternatives")
    route_selection_ratio: RouteSelectionRatio = field(rename="route-selection")


@click.group()
@click.option('--config-file', default='config.json')
@click.option('--debug/--no-debug', default=False)
@click.pass_context
def single_node_simulator_conf(ctx,
                               config_file,
                               debug):
    # ensure that ctx.obj exists and is a dict (in case `cli()` is called by means other than the `if` block bellow)
    ctx.ensure_object(dict)

    if os.path.isfile(config_file):
        logging.info(f"Settings taken from config file {config_file}.")
        with open(config_file, 'r') as f:
            config_data = f.read()
            args = from_json(Args, config_data)
    else:
        args = Args(CommonArgs(), RunArgs(), AlternativesRatio(), RouteSelectionRatio())

    ctx.obj['DEBUG'] = debug
    ctx.obj['common-args'] = args.common
    ctx.obj['run-args'] = args.run
    ctx.obj['alternatives-ratio'] = args.alternatives_ratio
    ctx.obj['route-selection-ratio'] = args.route_selection_ratio


@single_node_simulator_conf.command()
@click.pass_context
def run(ctx):
    common_args = ctx.obj["common-args"]
    run_args = ctx.obj["run-args"]
    alternatives_ratio = ctx.obj["alternatives-ratio"]
    route_selection_ratio = ctx.obj["route-selection-ratio"]
    p = Path(run_args.vehicles_path) if run_args.vehicles_path is not None else None
    run_inner(common_args, p, alternatives_ratio, route_selection_ratio)


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
