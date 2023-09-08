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
from ..tools.simulator import run_inner


@serde(rename_all="kebabcase", type_check=Strict)
@dataclass
class CommonArgs:
    task_id: Optional[str] = None
    departure_time: datetime = field(serializer=lambda x: x.strftime("%Y-%m-%d %H:%M:%S"),
                                     deserializer=lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"),
                                     default=datetime.now())
    round_frequency_s: timedelta = field(serializer=lambda x: x.total_seconds(),
                                         deserializer=lambda x: timedelta(seconds=x),
                                         default=timedelta(seconds=5))
    k_alternatives: int = 1
    map_update_freq_steps: int = 1
    count_vehicles_tolerance_s: timedelta = field(serializer=lambda x: x.total_seconds(),
                                                  deserializer=lambda x: timedelta(seconds=x),
                                                  default=timedelta(seconds=5))
    speeds_path: Optional[str] = None
    out: str = "simulation-record.pickle"
    seed: Optional[int] = None
    walltime_s: Optional[timedelta] = field(serializer=lambda x: x.total_seconds(),
                                            deserializer=lambda x: timedelta(seconds=x),
                                            default=None)
    saving_interval_s: Optional[timedelta] = field(serializer=lambda x: x.total_seconds(),
                                                   deserializer=lambda x: timedelta(seconds=x),
                                                   default=None)
    continue_from: Optional[Simulation] = field(serializer=lambda x: x.store("continue-from.pickle"),
                                                deserializer=lambda x: Simulation.load(x),
                                                default=None)


@serde(rename_all="kebabcase")
@dataclass
class RunArgs:
    vehicles_path: Optional[str] = None
    alternatives: Optional[str] = None
    route_selection: Optional[str] = None


@serde(rename_all="kebabcase")
@dataclass
class Args:
    common: CommonArgs = field(rename="ruth-simulator")
    run: RunArgs = field(rename="run")


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
        args = Args(CommonArgs(), RunArgs())

    ctx.obj['DEBUG'] = debug
    ctx.obj['common-args'] = args.common
    ctx.obj['run-args'] = args.run


@single_node_simulator_conf.command()
@click.pass_context
def run(ctx):
    common_args = ctx.obj["common-args"]
    run_args = ctx.obj["run-args"]
    p = Path(run_args.vehicles_path) if run_args.vehicles_path is not None else None
    run_inner(common_args, p, run_args.alternatives, run_args.route_selection)


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
