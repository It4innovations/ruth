
import click
from datetime import datetime, timedelta
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Optional

from probduration import HistoryHandler

from ..simulator import Simulation, SimSetting, SingleNodeSimulator, RouteRankingAlgorithms, load_vehicles
from ..losdb import FreeFlowDb, ProbProfileDb


@dataclass
class CommonArgs:
    task_id: str
    departure_time: datetime
    round_frequency: timedelta
    k_alternatives: int
    nproc: int
    out: str
    walltime: Optional[datetime] = None


@contextmanager
def prepare_simulator(common_args: CommonArgs, vehicles_path):
    departure_time = common_args.departure_time
    round_frequency = common_args.round_frequency
    k_alternatives = common_args.k_alternatives
    nproc = common_args.nproc

    # TODO: solve the debug symbol
    ss = SimSetting(departure_time, round_frequency, k_alternatives)
    vehicles = load_vehicles(vehicles_path)

    simulation = Simulation(vehicles, ss)

    with SingleNodeSimulator(simulation, nproc) as simulator:
        yield simulator


def store_simulation_at_walltime():
    saved = False

    def store(simulator: SingleNodeSimulator, walltime: timedelta, name: str):
        nonlocal saved
        """Store the state of the simulation at walltime."""
        if simulator.current_offset is not None and not saved and simulator.current_offset >= walltime:
            simulator.state.store(f"{name}-at-walltime.pickle")
            saved = True

    return store


@click.group()
@click.option('--debug/--no-debug', default=False)  # TODO: maybe move to top-level group
@click.option("--task-id", type=str)
@click.option("--departure-time", type=click.DateTime(), default=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))
@click.option("--round-frequency-s", type=int, default=5,
              help="Rounding time frequency in seconds.")
@click.option("--k-alternatives", type=int, default=1,
              help="Number of alternative routes.")
@click.option("--nproc", type=int, default=1,
              help="Number of concurrent processes.")
@click.option("--out", type=str, default="out.pickle")
@click.option("--walltime_s", type=int)
@click.pass_context
def single_node_simulator(ctx,
                          debug,
                          task_id,
                          departure_time,
                          round_frequency_s,
                          k_alternatives,
                          nproc,
                          out,
                          walltime_s):
    # ensure that ctx.obj exists and is a dict (in case `cli()` is called by means other than the `if` block bellow)
    ctx.ensure_object(dict)

    ctx.obj['DEBUG'] = debug
    walltime = timedelta(seconds=walltime_s) if walltime_s is not None else None
    sim_state = Simulation.load(continue_from) if continue_from is not None else None

    ctx.obj['common-args'] = CommonArgs(task_id,
                                        departure_time,
                                        timedelta(seconds=round_frequency_s),
                                        k_alternatives,
                                        nproc,
                                        out,
                                        walltime)


@single_node_simulator.command()
@click.argument("vehicles_path", type=click.Path(exists=True))
@click.pass_context
def rank_by_duration(ctx,
                     vehicles_path):

    common_args = ctx.obj['common-args']
    out = common_args.out
    walltime = common_args.walltime
    task_id = f"-task-{common_args.task_id}" if common_args.task_id is not None else ""

    with prepare_simulator(common_args, vehicles_path) as simulator:
        alg = RouteRankingAlgorithms.DURATION.value
        end_step_fn = store_simulation_at_walltime() if walltime is not None else lambda *_: None
        simulator.simulate(alg.rank_route, rr_fn_args=(simulator.state.global_view_db,),
                           end_step_fn=end_step_fn,
                           es_fn_args=(walltime, f"rank_by_duration{task_id}"))
        simulator.state.store(out)


@single_node_simulator.command()
@click.argument("vehicles_path", type=click.Path(exists=True))
@click.argument("prob_profile_path", type=click.Path(exists=True))
@click.argument("near_distance", type=float)
@click.argument("n_samples", type=int)
@click.pass_context
def rank_by_prob_delay(ctx,
                       vehicles_path,
                       prob_profile_path,
                       near_distance,
                       n_samples):

    """Perform the simulation on a cluster's single node. The simulation use for ranking alternative routes
    _probable delay_ on a route at a departure time. To compute the probable delay Monte Carlo Simulation is performed
    with N_SAMPLES iterations and the average delay is used. During the Monte Carlo simulation the speed on segments
    is changing according to probability profiles (PROB_PROFILE_PATH). The probability profile contains a distribution
    of _level of service_ at each segment which is valid for certain period (e.g., for 15minutes periods).
    NEAR_DISTANCE is used to compute the delay on route based on global view only. This can be seen as a distance
    "driver is seeing in front of her/him".
    """

    common_args = ctx.obj['common-args']
    out = common_args.out
    walltime = common_args.walltime
    task_id = f"-task-{common_args.task_id}" if common_args.task_id is not None else ""

    with prepare_simulator(ctx.obj['common-args'], vehicles_path) as simulator:
        alg = RouteRankingAlgorithms.PROBABLE_DELAY.value
        ff_db = FreeFlowDb()
        pp_db = ProbProfileDb(HistoryHandler.open(prob_profile_path))
        simulation = simulator.state
        end_step_fn = store_simulation_at_walltime() if walltime is not None else lambda *_: None
        simulator.simulate(alg.rank_route,
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


def main():
    single_node_simulator(obj={})


if __name__ == "__main__":
    main()
