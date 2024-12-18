import logging
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Type

import click

from ..flowmap.app import click_animation_options
from ..flowmap import animation

from ..vehicle import set_vehicle_behavior
from ..simulator import SimSetting, Simulation, SingleNodeSimulator, \
    load_vehicles
from ..simulator.kernels import AlternativesProvider, FastestPathsAlternatives, FirstRouteSelection, \
    RandomRouteSelection, RouteSelectionProvider, ShortestPathsAlternatives, \
    ZeroMQDistributedAlternatives, ZeroMQDistributedPTDRRouteSelection
from ..simulator.ptdr import PTDRInfo
from ..zeromq.src.client import Client


@dataclass
class CommonArgs:
    task_id: str
    departure_time: datetime
    round_frequency: timedelta
    k_alternatives: int
    map_update_freq: timedelta
    los_vehicles_tolerance: timedelta
    travel_time_limit_perc: float = 0.0
    speeds_path: Optional[str] = None
    out: str = "simulation-record.pickle"
    seed: Optional[int] = None
    walltime: Optional[timedelta] = None
    saving_interval: Optional[timedelta] = None
    continue_from: Optional[Simulation] = None
    stuck_detection: int = 0
    plateau_default_route: bool = False


@dataclass
class AlternativesRatio:
    default: float
    dijkstra_fastest: float
    dijkstra_shortest: float
    plateau_fastest: float

    def __post_init__(self):
        self.default = 1 - self.dijkstra_fastest - self.dijkstra_shortest - self.plateau_fastest
        self.default = round(self.default, 2)
        if self.default < 0:
            raise ValueError("Sum of alternatives ratios must be less than or equal to 1.")

    def to_list(self):
        return [self.default, self.dijkstra_fastest, self.dijkstra_shortest, self.plateau_fastest]


@dataclass
class RouteSelectionRatio:
    no_alternative: float
    first: float
    random: float
    ptdr: float

    def __post_init__(self):
        self.no_alternative = 1 - self.first - self.random - self.ptdr
        self.no_alternative = round(self.no_alternative, 2)
        if self.no_alternative < 0:
            raise ValueError("Sum of route selection ratios must be less than or equal to 1.")

    def to_list(self):
        return [self.no_alternative, self.first, self.random, self.ptdr]


def prepare_simulator(common_args: CommonArgs, vehicles_path, alternatives_ratio: AlternativesRatio,
                      route_selection_ratio: RouteSelectionRatio) \
        -> SingleNodeSimulator:
    departure_time = common_args.departure_time
    round_frequency = common_args.round_frequency
    k_alternatives = common_args.k_alternatives
    map_update_freq = common_args.map_update_freq
    los_vehicles_tolerance = common_args.los_vehicles_tolerance
    travel_time_limit_perc = common_args.travel_time_limit_perc
    seed = common_args.seed
    speeds_path = common_args.speeds_path
    sim_state = common_args.continue_from
    stuck_detection = common_args.stuck_detection
    plateau_default_route = common_args.plateau_default_route

    # TODO: solve the debug symbol
    if sim_state is None:
        if vehicles_path is None:
            raise ValueError("Either vehicles_path or continue_from must be specified.")
        ss = SimSetting(departure_time, round_frequency, k_alternatives, map_update_freq,
                        los_vehicles_tolerance, travel_time_limit_perc, seed, speeds_path=speeds_path,
                        stuck_detection=stuck_detection,
                        plateau_default_route=plateau_default_route)
        vehicles, bbox, download_date = load_vehicles(vehicles_path)

        set_vehicle_behavior(vehicles, alternatives_ratio.to_list(), route_selection_ratio.to_list())

        simulation = Simulation(vehicles, ss, bbox, download_date)
        if speeds_path is not None:
            simulation.routing_map.init_temporary_max_speeds(speeds_path)
    else:
        simulation = sim_state

    return SingleNodeSimulator(simulation)


def store_simulation_at_walltime(walltime: Optional[timedelta], name: str):
    saved = False
    start_time = datetime.now()

    def store(simulation: Simulation):
        nonlocal saved
        """Store the state of the simulation at walltime."""
        if walltime is not None and (datetime.now() - start_time) >= walltime and not saved:
            simulation.store(f"{name}-at-walltime.pickle")
            saved = True

    return store


def store_simulation_at_interval(saving_interval: Optional[timedelta], name: str):
    last_saved = datetime.now()

    def store(simulation: Simulation):
        nonlocal last_saved
        """Store the state of the simulation at interval."""
        now = datetime.now()
        if saving_interval is not None and (now - last_saved) >= saving_interval:
            simulation.store(f"{name}-interval-temp.pickle")
            last_saved = now
            shutil.move(f"{name}-interval-temp.pickle", f"{name}-interval.pickle")

    return store


def start_zeromq_cluster(
        worker_nodes: List[str],
        worker_per_node: int,
        map_path: Path,
        server_host: str = "localhost",
        port: int = 5559
):
    from cluster.cluster import Cluster, start_process

    ROOT_DIR = Path(__file__).absolute().parent.parent.parent
    WORKER_SCRIPT = ROOT_DIR / "ruth" / "zeromq" / "ex_worker.py"
    WORKER_DIR = ROOT_DIR / "worker-dir"

    VIRTUAL_ENV = os.environ["VIRTUAL_ENV"]

    WORKER_DIR.mkdir(parents=True, exist_ok=True)

    cluster = Cluster(str(WORKER_DIR))
    for worker_host in worker_nodes:
        for worker_index in range(worker_per_node):
            process = start_process(
                commands=[
                    sys.executable,
                    str(WORKER_SCRIPT),
                    "--address", server_host,
                    "--port", str(port),
                    "--map", str(map_path)
                ],
                workdir=str(WORKER_DIR),
                pyenv=VIRTUAL_ENV,
                hostname=worker_host,
                name=f"worker_{worker_host}_{worker_index}"
            )
            logging.info(f"Started worker {worker_host}:{process.pid}")
            cluster.add(process, key="worker")
    return cluster


@click.group(chain=True)
@click.option('--debug/--no-debug', default=False)  # TODO: maybe move to top-level group
@click.option("--task-id", type=str,
              help="A string to differentiate result outputs when two or more simulations are simultaneously running.")
@click.option("--departure-time", type=click.DateTime(),
              default=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
@click.option("--round-frequency-s", type=int, default=5,
              help="Interval (in seconds, of simulation time) for car selection to be moved in one simulation step. ")
@click.option("--k-alternatives", type=int, default=1,
              help="Number of alternative routes.")
@click.option("--map-update-freq-s", type=int, default=1,
              help="Frequency of updating map with current speeds (in seconds, in simulation time).")
@click.option("--los-vehicles-tolerance-s", type=int, default=1,
              help="Time tolerance (in seconds, of simulation time) to count which cars (i.e., their timestamps)"
                   "are considered for the calculation of LoS in a segment.")
@click.option("--travel-time-limit-perc", type=float, default=0.0)
@click.option("--speeds-path", type=click.Path(exists=True),
              help="Path to csv file with temporary max speeds.")
@click.option("--out", type=str, default="out.pickle")
@click.option("--seed", type=int, help="Fixed seed for random number generator.")
@click.option("--walltime-s", type=int, help="Time limit in which the state of simulation is saved")
@click.option("--saving-interval-s", type=int,
              help="Time interval in which the state of simulation periodically is saved")
@click.option("--continue-from", type=click.Path(exists=True),
              help="Path to a saved state of simulation to continue from.")
@click.option("--stuck-detection", type=int, default=0,
              help="Number of round-frequency-s long rounds with no vehicles movement before the simulation is "
                   "terminated. 0 means no stuck detection.")
@click.option("--plateau-default-route", is_flag=True,
              help="Recalculate default route to plateau fastest route.")
@click.pass_context
def single_node_simulator(ctx,
                          debug,
                          task_id,
                          departure_time,
                          round_frequency_s,
                          k_alternatives,
                          map_update_freq_s,
                          los_vehicles_tolerance_s,
                          travel_time_limit_perc,
                          speeds_path,
                          out,
                          seed,
                          walltime_s,
                          saving_interval_s,
                          continue_from,
                          stuck_detection,
                          plateau_default_route):
    # ensure that ctx.obj exists and is a dict (in case `cli()` is called by means other than the `if` block bellow)
    ctx.ensure_object(dict)

    ctx.obj['DEBUG'] = debug
    walltime = timedelta(seconds=walltime_s) if walltime_s is not None else None
    saving_interval = timedelta(
        seconds=saving_interval_s) if saving_interval_s is not None else None
    sim_state = Simulation.load(continue_from) if continue_from is not None else None

    ctx.obj['common-args'] = CommonArgs(
        task_id=task_id,
        departure_time=departure_time,
        round_frequency=timedelta(seconds=round_frequency_s),
        k_alternatives=k_alternatives,
        map_update_freq=timedelta(seconds=map_update_freq_s),
        los_vehicles_tolerance=timedelta(seconds=los_vehicles_tolerance_s),
        travel_time_limit_perc=travel_time_limit_perc,
        speeds_path=speeds_path,
        out=out,
        seed=seed,
        walltime=walltime,
        saving_interval=saving_interval,
        continue_from=sim_state,
        stuck_detection=stuck_detection,
        plateau_default_route=plateau_default_route,
    )


class ZeroMqContext:
    def __init__(self):
        self.clients = {}

    def get_or_create_client(self, port: int, broadcast_port: int) -> Client:
        if port not in self.clients:
            assert broadcast_port not in self.clients
            self.clients[port] = Client(port=port, broadcast_port=broadcast_port)
        return self.clients[port]


def create_alternatives_providers(alternatives_ratio: AlternativesRatio,
                                  zmq_ctx: ZeroMqContext,
                                  plateu_default_route: bool = False,
                                  ) -> List[AlternativesProvider]:
    providers = []
    if alternatives_ratio.dijkstra_fastest > 0:
        providers.append(FastestPathsAlternatives())
    if alternatives_ratio.dijkstra_shortest > 0:
        providers.append(ShortestPathsAlternatives())
    if alternatives_ratio.plateau_fastest > 0 or plateu_default_route:
        port = int(os.environ['port']) if 'port' in os.environ else 5555
        brodcast_port = int(os.environ['broadcast_port']) if 'broadcast_port' in os.environ else 5556
        providers.append(ZeroMQDistributedAlternatives(
            client=zmq_ctx.get_or_create_client(port=port, broadcast_port=brodcast_port)))
    return providers


def create_route_selection_providers(route_selection_ratio: RouteSelectionRatio,
                                     zeromq_ctx: ZeroMqContext,
                                     seed: Optional[int] = None) -> List[RouteSelectionProvider]:
    providers = []
    if route_selection_ratio.first > 0:
        providers.append(FirstRouteSelection())
    if route_selection_ratio.random > 0:
        providers.append(RandomRouteSelection(seed))
    if route_selection_ratio.ptdr > 0:
        providers.append(ZeroMQDistributedPTDRRouteSelection(client=zeromq_ctx.get_or_create_client(5555, 5556)))

    return providers


@single_node_simulator.command()
@click.argument("vehicles_path", type=click.Path(exists=True))
@click.option("--alt-dijkstra-fastest", type=float, default=0.0)
@click.option("--alt-dijkstra-shortest", type=float, default=0.0)
@click.option("--alt-plateau-fastest", type=float, default=0.0)
@click.option("--selection-first", type=float, default=0.0)
@click.option("--selection-random", type=float, default=0.0)
@click.option("--selection-ptdr", type=float, default=0.0)
@click.pass_context
def run(ctx,
        vehicles_path: Path,
        alt_dijkstra_fastest: float,
        alt_dijkstra_shortest: float,
        alt_plateau_fastest: float,
        selection_first: float,
        selection_random: float,
        selection_ptdr: float):
    common_args = ctx.obj["common-args"]

    alternatives_ratio = AlternativesRatio(
        default=0.0,
        dijkstra_fastest=alt_dijkstra_fastest,
        dijkstra_shortest=alt_dijkstra_shortest,
        plateau_fastest=alt_plateau_fastest
    )

    route_selection_ratio = RouteSelectionRatio(
        no_alternative=0.0,
        first=selection_first,
        random=selection_random,
        ptdr=selection_ptdr
    )

    ctx.obj['simulation'] = run_inner(common_args, vehicles_path, alternatives_ratio, route_selection_ratio)


def run_inner(common_args: CommonArgs, vehicles_path: Path,
              alternatives_ratio, route_selection_ratio) -> Simulation:
    out = common_args.out
    walltime = common_args.walltime
    saving_interval = common_args.saving_interval
    task_id = f"-task-{common_args.task_id}" if common_args.task_id is not None else ""

    simulator = prepare_simulator(common_args, vehicles_path, alternatives_ratio, route_selection_ratio)
    end_step_fns = []

    if walltime is not None:
        end_step_fns.append(store_simulation_at_walltime(walltime, f"run{task_id}"))

    if saving_interval is not None:
        end_step_fns.append(store_simulation_at_interval(saving_interval, f"run{task_id}"))

    zmq_ctx = ZeroMqContext()
    alternatives_providers = create_alternatives_providers(alternatives_ratio,
                                                           zmq_ctx=zmq_ctx,
                                                           plateu_default_route=common_args.plateau_default_route)
    route_selection_providers = create_route_selection_providers(route_selection_ratio, zeromq_ctx=zmq_ctx,
                                                                 seed=common_args.seed)

    for route_selection_provider in route_selection_providers:
        if isinstance(route_selection_provider, ZeroMQDistributedPTDRRouteSelection):
            ptdr_info = PTDRInfo(common_args.departure_time)
            route_selection_provider.update_segment_profiles(ptdr_info)
            break

    simulator.simulate(
        alternatives_providers=alternatives_providers,
        route_selection_providers=route_selection_providers,
        end_step_fns=end_step_fns,
    )
    simulation = simulator.state
    simulation.store(out)
    return simulation


def animate(ctx, animator: Type[animation.SimulationAnimator], **kwargs):
    # check if simulation has been run and whether it finished
    if not ctx.obj.get('simulation'):
        raise click.ClickException("The 'animation' command can only be used after 'run' command.")

    simulation = ctx.obj['simulation']
    simulation_finished = simulation.finished()

    if not simulation_finished:
        logging.warning("Creating animation of an unfinished simulation.")

    animation_creator = animator(simulation_path=ctx.obj['common-args'].out, **kwargs)
    animation_creator.set_simulation(simulation)
    animation_creator.run()


@single_node_simulator.command()
@click_animation_options
@click.pass_context
def volume_animation(ctx, **kwargs):
    animate(ctx, animation.SimulationVolumeAnimator, **kwargs)


@single_node_simulator.command()
@click_animation_options
@click.pass_context
def speed_animation(ctx, **kwargs):
    animate(ctx, animation.SimulationSpeedsAnimator, **kwargs)


def main():
    log_level = logging.INFO
    log_level_env = os.environ.get("LOG_LEVEL", "")
    if log_level_env.lower() == "debug":
        log_level = logging.DEBUG

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(name)s:%(levelname)-4s %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
        force=True
    )
    single_node_simulator(obj={})


if __name__ == "__main__":
    main()
