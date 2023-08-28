import enum
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import click
from probduration import HistoryHandler

from ruth.losdb import FreeFlowDb, ProbProfileDb
from ruth.simulator import RouteRankingAlgorithms, SimSetting, Simulation, SingleNodeSimulator, \
    load_vehicles
from ruth.simulator.kernels import AlternativesProvider, FastestPathsAlternatives, \
    FirstRouteSelection, RouteSelectionProvider, ShortestPathsAlternatives, \
    ZeroMQDistributedAlternatives, RandomRouteSelection


@dataclass
class CommonArgs:
    task_id: str
    departure_time: datetime
    round_frequency: timedelta
    k_alternatives: int
    map_update_freq_steps: int
    count_vehicles_tolerance: timedelta
    speeds_path: str
    out: str
    seed: Optional[int] = None
    walltime: Optional[datetime] = None
    saving_interval: Optional[datetime] = None
    continue_from: Optional[Simulation] = None


def prepare_simulator(common_args: CommonArgs, vehicles_path) -> SingleNodeSimulator:
    departure_time = common_args.departure_time
    round_frequency = common_args.round_frequency
    k_alternatives = common_args.k_alternatives
    map_update_freq_steps = common_args.map_update_freq_steps
    count_vehicles_tolerance = common_args.count_vehicles_tolerance
    speeds_path = common_args.speeds_path
    seed = common_args.seed
    sim_state = common_args.continue_from

    # TODO: solve the debug symbol
    if sim_state is None:
        ss = SimSetting(departure_time, round_frequency, k_alternatives, map_update_freq_steps, count_vehicles_tolerance,
                        seed, speeds_path)
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
            if os.path.isfile(f"{name}-interval.pickle"):
                os.remove(f"{name}-interval.pickle")
            os.rename(f"{name}-interval-temp.pickle", f"{name}-interval.pickle")

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
@click.option('--config-file', default='config.json')
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
@click.option("--map-update-freq-steps", type=int, default=1,
              help="Step frequency of changing the map with current speeds.")
@click.option("--count-vehicles-tolerance", type=int, default=1,
              help="Time tolerance in seconds for counting cars on a segment for LoS.")
@click.option("--speeds-path", type=click.Path(exists=True))
@click.option("--out", type=str, default="out.pickle")
@click.option("--seed", type=int, help="Fixed seed for random number generator.")
@click.option("--walltime-s", type=int, help="Time limit in which the state of simulation is saved")
@click.option("--saving-interval-s", type=int, help="Time interval in which the state of simulation periodically is saved")
@click.option("--continue-from", type=click.Path(exists=True),
              help="Path to a saved state of simulation to continue from.")
@click.pass_context
def single_node_simulator(ctx,
                          config_file,
                          debug,
                          task_id,
                          departure_time,
                          round_frequency_s,
                          k_alternatives,
                          map_update_freq_steps,
                          count_vehicles_tolerance,
                          speeds_path,
                          out,
                          seed,
                          walltime_s,
                          saving_interval_s,
                          continue_from):
    # ensure that ctx.obj exists and is a dict (in case `cli()` is called by means other than the `if` block bellow)
    ctx.ensure_object(dict)

    if os.path.isfile(config_file):
        logging.info(f"Settings taken from config file {config_file}.")
        with open(config_file) as f:
            config_data = json.load(f)['ruth-simulator']

            debug = config_data.get('debug', debug)
            task_id = config_data.get('task-id', task_id)
            departure_time = config_data.get('departure-time', departure_time)
            if type(departure_time) == str:
                departure_time = datetime.strptime(departure_time, "%Y-%m-%d %H:%M:%S")

            round_frequency_s = config_data.get('round-frequency-s', round_frequency_s)
            k_alternatives = config_data.get('k-alternatives', k_alternatives)
            map_update_freq_steps = config_data.get('map-update-freq-steps', map_update_freq_steps)
            count_vehicles_tolerance = config_data.get('count-vehicles-tolerance', count_vehicles_tolerance)
            speeds_path_config = config_data.get('speeds-path', None)
            if speeds_path_config is not None:
                speeds_path = Path(speeds_path_config)
            out = config_data.get('out', out)
            seed = config_data.get('seed', seed)
            walltime_s = config_data.get('walltime-s', walltime_s)
            saving_interval_s = config_data.get('saving-interval-s', saving_interval_s)
            continue_from_config = config_data.get('continue-from', None)
            if continue_from_config is not None:
                continue_from = Path(continue_from_config)

    ctx.obj['DEBUG'] = debug
    ctx.obj['config-file-path'] = config_file

    walltime = timedelta(seconds=walltime_s) if walltime_s is not None else None
    saving_interval = timedelta(seconds=saving_interval_s) if saving_interval_s is not None else None
    sim_state = Simulation.load(continue_from) if continue_from is not None else None

    ctx.obj['common-args'] = CommonArgs(task_id,
                                        departure_time,
                                        timedelta(seconds=round_frequency_s),
                                        k_alternatives,
                                        map_update_freq_steps,
                                        timedelta(seconds=count_vehicles_tolerance),
                                        speeds_path,
                                        out,
                                        seed,
                                        walltime,
                                        saving_interval,
                                        sim_state)


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
        # TODO: parse port from CLI
        return ZeroMQDistributedAlternatives(port=5555)
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


@single_node_simulator.command()
@click.option("--vehicles-path", type=click.Path(exists=True))
@click.option("--alternatives", type=click.Choice(AlternativesImpl),
              default=AlternativesImpl.FASTEST_PATHS)
@click.option("--route-selection", type=click.Choice(RouteSelectionImpl),
              default=RouteSelectionImpl.RANDOM)
@click.pass_context
def run(ctx,
        vehicles_path: Path,
        alternatives: AlternativesImpl,
        route_selection: RouteSelectionImpl):

    config_file = ctx.obj["config-file-path"]
    if os.path.isfile(config_file):
        with open(config_file) as f:
            config_data = json.load(f)['run']
            vehicles_path_config = config_data.get('vehicles-path', None)
            if vehicles_path_config is not None:
                vehicles_path = Path(vehicles_path_config)
            alternatives_config = config_data.get('alternatives', None)
            if alternatives_config is not None:
                alternatives = AlternativesImpl(alternatives_config)
            route_selection_config = config_data.get('route-selection', None)
            if route_selection_config is not None:
                route_selection = RouteSelectionImpl(route_selection_config)

    if vehicles_path is None:
        logging.error("Vehicle path has to be set.")
        return 1

    common_args = ctx.obj["common-args"]
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s:%(levelname)-4s %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
        force=True
    )
    single_node_simulator(obj={})


if __name__ == "__main__":
    main()
