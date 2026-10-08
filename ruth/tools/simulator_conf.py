import json
import logging
import os
import sys
from dataclasses import dataclass, make_dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import click
from click import IntRange
from serde import serde, field
from serde.json import from_json


def configure_logging():
    """Configure logging early, before any imports that might use it."""
    log_level = logging.INFO
    if os.environ.get("LOG_LEVEL", "").lower() == "debug":
        log_level = logging.DEBUG

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(name)s:%(levelname)-4s %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
        force=True,
        stream=sys.stdout,
    )


configure_logging()

from ..tools.simulator import (run_inner, AlternativesRatio as AlternativesRatioInner, CommonArgs as CommonArgsInner,
                               RouteSelectionRatio as RouteSelectionRatioInner, AceArgs as AceArgsInner,
                               ConsoleArgs as ConsoleArgsInner)


def serialize_optional_timedelta(value: Optional[timedelta]):
    return None if value is None else value.total_seconds()


def deserialize_optional_timedelta(value):
    return None if value is None else timedelta(seconds=value)


def make_animation_args_dataclass(options_dict: dict) -> type:
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


@serde(rename_all="kebabcase")
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
    continue_from: str = ""
    stuck_detection: int = 0
    plateau_default_route: bool = False
    buffer_size: int = 10_000
    max_records_per_file: int = int(1e9)
    max_steps: Optional[int] = field(rename="max-steps", default=None)
    vehicle_frequency_override: Optional[timedelta] = field(rename="vehicle-frequency-override-s",
                                                           serializer=serialize_optional_timedelta,
                                                           deserializer=deserialize_optional_timedelta,
                                                           default=None)
    fcd_sampling_period_override: Optional[timedelta] = field(rename="fcd-sampling-period-override-s",
                                                             serializer=serialize_optional_timedelta,
                                                             deserializer=deserialize_optional_timedelta,
                                                             default=None)


    async_fcd_writer: bool = False
    fcd_writer_queue_size: int = 4
    map_graphml: Optional[str] = field(rename="map-graphml", default=None)
    overwrite_h5: bool = field(rename="overwrite-h5", default=False)
    max_vehicles: Optional[int] = field(rename="max-vehicles", default=None)


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
class AceArgs(AceArgsInner):
    # kebab-case JSON keys: "threads", "mpi-advance", "queue",
    # "deterministic-order", "trajectory-digest", "trajectory-dump-step".
    # threads=None -> all cores.
    threads: Optional[int] = None
    mpi_advance: bool = False
    queue: str = "prioritized"
    deterministic_order: bool = False
    trajectory_digest: bool = False
    trajectory_dump_step: int = -1


@serde(rename_all="kebabcase")
@dataclass
class ConsoleArgs(ConsoleArgsInner):
    # kebab-case JSON keys: "enabled", "host", "port", "snapshot-every-steps",
    # "top-segments", "max-vehicles-in-snapshot".
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8787
    snapshot_every_steps: int = 1
    top_segments: int = 20
    max_vehicles_in_snapshot: int = 2000


@serde(rename_all="kebabcase")
@dataclass
class Args:
    common: CommonArgs = field(rename="ruth-simulator")
    run: RunArgs = field(rename="run")
    alternatives_ratio: AlternativesRatio = field(rename="alternatives")
    route_selection_ratio: RouteSelectionRatio = field(rename="route-selection")
    # Optional section; None when "ace" is absent (run_inner falls back to defaults).
    ace: Optional[AceArgs] = field(rename="ace", default=None)
    # Optional section; None when "console" is absent (console stays disabled).
    console: Optional[ConsoleArgs] = field(rename="console", default=None)


def fill_args(config_file: str, ctx: Optional[click.Context] = None, workdir: Optional[str] = None, debug: bool = False) -> tuple[Args, Path]:
    if os.path.isfile(config_file):
        # Log the ABSOLUTE path: --config-file defaults to a CWD-relative name, so
        # running from a different folder can silently read a different config.
        logging.info(f"Settings taken from config file {Path(config_file).absolute()}.")
        with open(config_file, 'r') as f:
            config_data = f.read()
            args = from_json(Args, config_data)
            config_json = json.loads(config_data)  # raw view for the map-graphml guard
    else:
        logging.info(f"Config file not found.")
        args = Args(CommonArgs(), RunArgs(), AlternativesRatio(), RouteSelectionRatio())
        config_json = None

    p = args.run.vehicles_path
    if p is None:
        raise ValueError(f"Vehicles path not set. Config file: {Path(config_file).absolute()}")

    if workdir is not None:
        # p is relative to the workdir
        p = Path(workdir) / p
    else:
        p = Path(p)

    # check if vehicles path is valid
    if not p.exists():
        raise ValueError(f"Vehicles path {p.absolute()} does not exist.")

    # --- map-graphml: fail loudly instead of silently rebuilding a bbox map ---
    # When common.map_graphml is None the routing map is silently rebuilt from the
    # vehicles' bbox (Map._load / OSM download), which masks two mistakes:
    #   (a) map-graphml set but placed OUTSIDE the "ruth-simulator" object (pyserde
    #       drops unknown keys), and
    #   (b) map-graphml pointing at a missing/inaccessible file.
    # Both previously surfaced only as "No link ... in the network" / "No active
    # vehicles found". Catch them here at config time.
    def _find_key(obj, key):
        if isinstance(obj, dict):
            if key in obj:
                return obj[key]
            for value in obj.values():
                found = _find_key(value, key)
                if found is not None:
                    return found
        elif isinstance(obj, list):
            for item in obj:
                found = _find_key(item, key)
                if found is not None:
                    return found
        return None

    parsed_map_graphml = getattr(args.common, "map_graphml", None)
    raw_map_graphml = _find_key(config_json, "map-graphml") if config_json is not None else None
    if raw_map_graphml and not parsed_map_graphml:
        raise ValueError(
            f"'map-graphml' is present in {Path(config_file).absolute()} (value "
            f"{raw_map_graphml!r}) but was not applied. It must be nested INSIDE the "
            f"\"ruth-simulator\" object; a key placed anywhere else is silently ignored."
        )
    if parsed_map_graphml:
        mg = Path(parsed_map_graphml)
        if not mg.is_absolute() and workdir is not None:
            mg = Path(workdir) / mg
        if not mg.exists():
            raise ValueError(
                f"map-graphml is set to '{mg.absolute()}' but that file does not exist "
                f"(config: {Path(config_file).absolute()}). Refusing to fall back to a "
                f"bbox-derived map, which would mismatch the vehicles' routes."
            )
        # Pin the resolved absolute path so the loader can't reinterpret it per-CWD.
        args.common.map_graphml = str(mg.absolute())
        logging.info(f"Using map-graphml: {args.common.map_graphml}")

    if ctx is not None:
        ctx.obj['DEBUG'] = debug
        ctx.obj['common-args'] = args.common
        ctx.obj['run-args'] = args.run
        ctx.obj['alternatives-ratio'] = args.alternatives_ratio
        ctx.obj['route-selection-ratio'] = args.route_selection_ratio
        ctx.obj['ace-args'] = args.ace
        ctx.obj['console-args'] = args.console
        ctx.obj['path'] = p

    return args, p


@click.group(chain=True)
@click.option('--config-file', default='config.json')
@click.option('--workdir', default=None)
@click.option('--debug/--no-debug', default=False)
@click.pass_context
def single_node_simulator_conf(ctx,
                               config_file,
                               workdir,
                               debug):
    # ensure that ctx.obj exists and is a dict (in case `cli()` is called by means other than the `if` block bellow)
    ctx.ensure_object(dict)
    fill_args(config_file, ctx, workdir, debug)


@single_node_simulator_conf.command()
@click.pass_context
def run(ctx):
    common_args = ctx.obj["common-args"]
    run_args = ctx.obj["run-args"]
    alternatives_ratio = ctx.obj["alternatives-ratio"]
    route_selection_ratio = ctx.obj["route-selection-ratio"]
    ace_args = ctx.obj["ace-args"]
    console_args = ctx.obj["console-args"]
    p = ctx.obj["path"]

    ctx.obj['simulation'] = run_inner(common_args, p, alternatives_ratio, route_selection_ratio, ace_args,
                                      console_args)


def main():
    single_node_simulator_conf(obj={})


if __name__ == "__main__":
    main()
