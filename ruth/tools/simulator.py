import enum
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import click
from probduration import HistoryHandler

from ..losdb import FreeFlowDb, ProbProfileDb
from ..simulator import RouteRankingAlgorithms, SimSetting, Simulation, SingleNodeSimulator, \
    load_vehicles
from ..simulator.kernels import AlternativesProvider, FastestPathsAlternatives, \
    FirstRouteSelection, RouteSelectionProvider, ShortestPathsAlternatives, \
    ZeroMQDistributedAlternatives, RandomRouteSelection


@dataclass
class CommonArgs:
    task_id: str
    departure_time: datetime
    round_frequency: timedelta
    k_alternatives: int
    map_update_freq: timedelta
    count_vehicles_tolerance: timedelta
    speeds_path: Optional[str] = None
    out: str = "simulation-record.pickle"
    seed: Optional[int] = None
    walltime: Optional[timedelta] = None
    saving_interval: Optional[timedelta] = None
    continue_from: Optional[Simulation] = None


def prepare_simulator(common_args: CommonArgs, vehicles_path) -> SingleNodeSimulator:
    departure_time = common_args.departure_time
    round_frequency = common_args.round_frequency
    k_alternatives = common_args.k_alternatives
    map_update_freq = common_args.map_update_freq
    count_vehicles_tolerance_s = common_args.count_vehicles_tolerance
    seed = common_args.seed
    speeds_path = common_args.speeds_path
    sim_state = common_args.continue_from

    # TODO: solve the debug symbol
    if sim_state is None:
        if vehicles_path is None:
            raise ValueError("Either vehicles_path or continue_from must be specified.")
        ss = SimSetting(departure_time, round_frequency, k_alternatives, map_update_freq,
                        count_vehicles_tolerance_s, seed, speeds_path=speeds_path)
        vehicles = load_vehicles(vehicles_path)
        simulation = Simulation(vehicles, ss)
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


@click.group()
@click.option('--debug/--no-debug', default=False)  # TODO: maybe move to top-level group
@click.option("--task-id", type=str,
              help="A string to differentiate results if there is running more simulations"
                   " simultaneously.")
@click.option("--departure-time", type=click.DateTime(),
              default=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
@click.option("--round-frequency-s", type=int, default=5,
              help="Rounding time frequency in seconds.")
@click.option("--k-alternatives", type=int, default=1,
              help="Number of alternative routes.")
@click.option("--map-update-freq-s", type=int, default=1,
              help="Frequency in simulation time of changing the map with current speeds.")
@click.option("--count-vehicles-tolerance-s", type=int, default=1,
              help="Time tolerance in seconds for counting cars on a segment for LoS.")
@click.option("--speeds-path", type=click.Path(exists=True))
@click.option("--out", type=str, default="out.pickle")
@click.option("--seed", type=int, help="Fixed seed for random number generator.")
@click.option("--walltime-s", type=int, help="Time limit in which the state of simulation is saved")
@click.option("--saving-interval-s", type=int,
              help="Time interval in which the state of simulation periodically is saved")
@click.option("--continue-from", type=click.Path(exists=True),
              help="Path to a saved state of simulation to continue from.")
@click.pass_context
def single_node_simulator(ctx,
                          debug,
                          task_id,
                          departure_time,
                          round_frequency_s,
                          k_alternatives,
                          map_update_freq_s,
                          count_vehicles_tolerance_s,
                          speeds_path,
                          out,
                          seed,
                          walltime_s,
                          saving_interval_s,
                          continue_from):
    # ensure that ctx.obj exists and is a dict (in case `cli()` is called by means other than the `if` block bellow)
    ctx.ensure_object(dict)

    ctx.obj['DEBUG'] = debug
    walltime = timedelta(seconds=walltime_s) if walltime_s is not None else None
    saving_interval = timedelta(seconds=saving_interval_s) if saving_interval_s is not None else None
    sim_state = Simulation.load(continue_from) if continue_from is not None else None

    ctx.obj['common-args'] = CommonArgs(
                                    task_id=task_id,
                                    departure_time=departure_time,
                                    round_frequency=timedelta(seconds=round_frequency_s),
                                    k_alternatives=k_alternatives,
                                    map_update_freq=timedelta(seconds=map_update_freq_s),
                                    count_vehicles_tolerance=timedelta(seconds=count_vehicles_tolerance_s),
                                    speeds_path=speeds_path,
                                    out=out,
                                    seed=seed,
                                    walltime=walltime,
                                    saving_interval=saving_interval,
                                    continue_from=sim_state
                                )


@single_node_simulator.command()
@click.argument("vehicles_path", type=click.Path(exists=True))
@click.pass_context
def rank_by_duration(ctx,
                     vehicles_path):
    raise NotImplementedError
    common_args = ctx.obj['common-args']
    out = common_args.out
    walltime = common_args.walltime
    task_id = f"-task-{common_args.task_id}" if common_args.task_id is not None else ""

    simulator = prepare_simulator(common_args, vehicles_path)
    alg = RouteRankingAlgorithms.DURATION.value
    end_step_fn = store_simulation_at_walltime() if walltime is not None else lambda *_: None
    simulator.simulate(alg.rank_route, rr_fn_args=(simulator.state.global_view_db,),
                       end_step_fn=end_step_fn,
                       es_fn_args=(walltime, f"rank_by_duration{task_id}"))
    simulator.state.store(out)


@single_node_simulator.command()
@click.argument("vehicles_path", type=click.Path(exists=True))
@click.argument("near_distance", type=float)
@click.argument("n_samples", type=int)
@click.option("--prob_profile_path", type=click.Path(exists=True),
              help="A path to probability profile [Default: no limit prob. profile]")
@click.pass_context
def rank_by_prob_delay(ctx,
                       vehicles_path,
                       near_distance,
                       n_samples,
                       prob_profile_path):
    """Perform the simulation on a cluster's single node. The simulation use for ranking alternative routes
    _probable delay_ on a route at a departure time. To compute the probable delay Monte Carlo Simulation is performed
    with N_SAMPLES iterations and the average delay is used. During the Monte Carlo simulation the speed on segments
    is changing according to probability profiles (PROB_PROFILE_PATH). The probability profile contains a distribution
    of _level of service_ at each segment which is valid for certain period (e.g., for 15minutes periods).
    NEAR_DISTANCE is used to compute the delay on route based on global view only. This can be seen as a distance
    "driver is seeing in front of her/him".
    """

    raise NotImplementedError
    common_args = ctx.obj['common-args']
    out = common_args.out
    walltime = common_args.walltime
    task_id = f"-task-{common_args.task_id}" if common_args.task_id is not None else ""

    data_loading_start = datetime.now()
    simulator = prepare_simulator(ctx.obj['common-args'], vehicles_path)
    alg = RouteRankingAlgorithms.PROBABLE_DELAY.value
    ff_db = FreeFlowDb()
    if prob_profile_path == None:
        pp_db = ProbProfileDb(HistoryHandler.no_limit())
    else:
        pp_db = ProbProfileDb(HistoryHandler.open(prob_profile_path))
    simulation = simulator.state
    time_for_data_loading = datetime.now() - data_loading_start
    end_step_fn = store_simulation_at_walltime() if walltime is not None else lambda *_: None
    walltime = walltime - time_for_data_loading if walltime is not None else None

    zeromq = False
    if zeromq:
        # TODO: refactor this, remove singleton from Map
        map_path = Path(simulation.routing_map.file_path).absolute()
        server_port = 5559
        cluster = start_zeromq_cluster(
            worker_nodes=["localhost"],
            worker_per_node=4,
            map_path=map_path,
            server_host="127.0.0.1",
            port=server_port
        )
        kernel_provider = ZeroMQDistributedAlternatives(port=server_port)
    else:
        kernel_provider = ShortestPathsAlternatives()

    try:
        simulator.simulate(alg.rank_route,
                           alternatives_provider=kernel_provider,
                           extend_plans_fn=alg.prepare_data,
                           ep_fn_args=(simulation.global_view_db,
                                       near_distance,
                                       ff_db,
                                       pp_db,
                                       n_samples,
                                       simulation.setting.rnd_gen),
                           end_step_fn=end_step_fn,
                           es_fn_args=(walltime, f"rank-by-prob-profile{task_id}"))

        simulation.store(out)
    finally:
        if zeromq:
            cluster.kill()


@enum.unique
class AlternativesImpl(str, enum.Enum):
    FASTEST_PATHS = "fastest-paths"
    SHORTEST_PATHS = "shortest-paths"
    DISTRIBUTED = "distributed"


def create_alternatives_provider(alternatives: AlternativesImpl) -> AlternativesProvider:
    if alternatives == AlternativesImpl.FASTEST_PATHS:
        return FastestPathsAlternatives()
    elif alternatives == AlternativesImpl.SHORTEST_PATHS:
        return ShortestPathsAlternatives()
    elif alternatives == AlternativesImpl.DISTRIBUTED:
        from ..zeromq.src.client import Client
        client = Client(port=5555)
        return ZeroMQDistributedAlternatives(client=client)
    else:
        raise NotImplementedError


@enum.unique
class RouteSelectionImpl(str, enum.Enum):
    FIRST = "first"
    RANDOM = "random"


def create_route_selection_provider(route_selection: RouteSelectionImpl) -> RouteSelectionProvider:
    if route_selection == RouteSelectionImpl.FIRST:
        return FirstRouteSelection()
    elif route_selection == RouteSelectionImpl.RANDOM:
        return RandomRouteSelection()
    else:
        raise NotImplementedError
        print(route_selection)


@single_node_simulator.command()
@click.argument("vehicles_path", type=click.Path(exists=True))
@click.option("--alternatives", type=click.Choice(AlternativesImpl),
              default=AlternativesImpl.FASTEST_PATHS)
@click.option("--route-selection", type=click.Choice(RouteSelectionImpl),
              default=RouteSelectionImpl.FIRST)
@click.pass_context
def run(ctx,
        vehicles_path: Path,
        alternatives: AlternativesImpl,
        route_selection: RouteSelectionImpl):
    common_args = ctx.obj["common-args"]
    run_inner(common_args, vehicles_path, alternatives, route_selection)


def run_inner(common_args: CommonArgs, vehicles_path: Path, alternatives: AlternativesImpl,
              route_selection: RouteSelectionImpl):
    out = common_args.out
    walltime = common_args.walltime
    saving_interval = common_args.saving_interval
    task_id = f"-task-{common_args.task_id}" if common_args.task_id is not None else ""

    simulator = prepare_simulator(common_args, vehicles_path)
    end_step_fns = []

    if walltime is not None:
        end_step_fns.append(store_simulation_at_walltime(walltime, f"run{task_id}"))

    if saving_interval is not None:
        end_step_fns.append(store_simulation_at_interval(saving_interval, f"run{task_id}"))

    alternatives_provider = create_alternatives_provider(alternatives)
    route_selection_provider = create_route_selection_provider(route_selection)
    simulator.simulate(
        alternatives_provider=alternatives_provider,
        route_selection_provider=route_selection_provider,
        end_step_fns=end_step_fns,
    )
    simulator.state.store(out)


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
