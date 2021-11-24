import os
import click
from datetime import datetime
from cluster import cluster

from ruth.distsim import simulate


def dask_simulator(input_benchmark_data,
                   departure_time,
                   k_routes,
                   n_samples,
                   seed,
                   gv_update_period,
                   dask_scheduler,
                   dask_scheduler_port,
                   dask_workers,
                   out,
                   intermediate_results,
                   checkpoint_period,
                   pyenv):
    """Run the vehicle simulator in dask environment.

    Parameters:
    -----------
    input_benchmark_data: Path
        An input file with benchmark data.
    departure_time: datetime
        A departure time (in format: '%Y-%m-%dT%H:%M:%S') of vehicles. Each vehicle has its
        own `start_offset` that with combination of departure time determines the starting
        time of the vehicle.
    k_routes: int
        A number of alternative routes routed between two points. Each route is ranked and the
        _best rank_ is used for the vehicle.
    n_samples: int
        A number of samples of Monte Carlo simulation.
    seed: Optional[int]
        A fixed seed used for random generator.
    gv_update_period: int
        Global view update period; a number of consecutive steps perfomed between
        the update of the global view.
    dask_scheduler: str
        An address of the machine where the _dask scheduler_ is running.
    dask_scheduler_port: int
        A port of running _dask scheduler_.
    dask_workers: List[str]
        A list of addresses of machines where _dask workers_ are running.
    out: Path
        A path of output file.
    intermediate_results: Optional[Path]
        A path of directory where the intermediate results are stored. If `None` the intermediate
        results are not stored.
    checkpoint_period: int
        A period in which the intermediate results are stored.
    pyenv: Optional[Path]
        A path of directory with virtual enviroment where the package is installed.
        If `None` it is supposed that the package is installed in the user space.
    """

    cluster.start_process(["dask-scheduler",
                           "--port",
                           dask_scheduler_port],
                          host=dask_scheduler,
                          pyenv=pyenv)

    for worker in dask_workers:
        cluster.start_process(["dask-worker",
                               f"{dask_scheduler}:{dask_scheduler_port}"],
                              host=worker,
                              pyenv=pyenv)

    final_state_df = simulate(input_benchmark_data,
                              len(dask_workers),
                              departure_time,
                              k_routes,
                              n_samples,
                              seed,
                              gv_update_period,
                              dask_scheduler,
                              dask_scheduler_port,
                              intermediate_results,
                              checkpoint_period)

    out_path = os.path.abspath(out)
    final_state_df.to_pickle(out_path)


run_simulator = click.Group()

@run_simulator.command()
@click.argument("input_benchmark_data",
                type=click.Path(exists=True))
@click.option("--departure-time",
              type=click.DateTime(),
              default=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))
@click.option("--k-routes",
              type=int,
              default=4,
              help="A number of alterantive routes routed between two points.")
@click.option("--n-samples",
              type=int,
              default=1000,
              help="A number of samples of monte carlo simulation.")
@click.option("--seed",
              type=int,
              help=("A fixed seed for random number generator"
                    "enusiring the same behaviour in the next run."))
@click.option("--gv-update-period",
              type=int,
              default=10,
              help="An 'n' consecutive steps performed between update of the global view.")
@click.option("--dask-scheduler",
              type=str,
              default="localhost",
              help="An address of dask scheduler.")
@click.option("--dask-scheduler-port",
              type=int,
              default=8786,
              help="A port of the dask scheduler.")
@click.option("--dask-workers", multiple=True, default=["localhost"])
@click.option("--out", type=str, default="out.pickle")
@click.option("--intermediate-results",
              type=click.Path(),
              help="A path to the folder with intermediate results.")
@click.option("--checkpoint-period",
              type=int,
              default=1,
              help="A period in which the intermediate results are stored.")
@click.option("--pyenv",
              envvar="RUTH_PYENV",
              type=click.Path(exists=True),
              help=("An optional path to the virtual enviroment with installed packages."
                    "It can be also specified via the RUTH_PYENV environment variable."))
def generic(input_benchmark_data,
            departure_time,
            k_routes,
            n_samples,
            seed,
            gv_update_period,
            dask_scheduler,
            dask_scheduler_port,
            dask_workers,
            out,
            intermediate_results,
            checkpoint_period,
            pyenv):
    """Generic command calling the simulator within dask environment."""

    dask_simulator(input_benchmark_data,
                   departure_time,
                   k_routes,
                   n_samples,
                   seed,
                   gv_update_period,
                   dask_scheduler,
                   dask_scheduler_port,
                   dask_workers,
                   out,
                   intermediate_results,
                   checkpoint_period,
                   os.path.abspath(pyenv))


@run_simulator.command()
@click.argument("input_benchmark_data",
                type=click.Path(exists=True))
@click.option("--departure-time",
              type=click.DateTime(),
              default=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))
@click.option("--k-routes",
              type=int,
              default=4,
              help="A number of alterantive routes routed between two points.")
@click.option("--n-samples",
              type=int,
              default=1000,
              help="A number of samples of monte carlo simulation.")
@click.option("--seed",
              type=int,
              help=("A fixed seed for random number generator"
                    "enusiring the same behaviour in the next run."))
@click.option("--gv-update-period",
              type=int,
              default=10,
              help="An 'n' consecutive steps performed between update of the global view.")
@click.option("--dask-scheduler-port",
              type=int,
              default=8786,
              help="A port of the dask scheduler.")
@click.option("--out", type=str, default="out.pickle")
@click.option("--intermediate-results",
              type=click.Path(),
              help="A path to the folder with intermediate results.")
@click.option("--checkpoint-period",
              type=int,
              default=1,
              help="A period in which the intermediate results are stored.")
@click.option("--pyenv",
              envvar="RUTH_PYENV",
              type=click.Path(exists=True),
              help=("An optional path to the virtual enviroment with installed packages."
                    "It can be also specified via the RUTH_PYENV environment variable."))
def pbs(input_benchmark_data,
        departure_time,
        k_routes,
        n_samples,
        seed,
        gv_update_period,
        dask_scheduler_port,
        out,
        intermediate_results,
        checkpoint_period,
        pyenv):
    """PBS specific command calling the simulator within dask environment.

    This command load the info from PBS_NODEFILE and run the scheduler on the first node while
    the workers are launech on the rest of nodes.
    """

    pbs_nodefile = os.getenv("PBS_NODEFILE")
    print("pbs: ", pbs_nodefile)

    with open(pbs_nodefile) as f:
        hosts = f.readlines()
        nodes = [host.strip() for host in hosts]

    assert len(nodes) >= 2, "The PBS version must run on at least two nodes."

    dask_scheduler = nodes[0]
    dask_workers = nodes[1:]

    dask_simulator(input_benchmark_data,
                   departure_time,
                   k_routes,
                   n_samples,
                   seed,
                   gv_update_period,
                   dask_scheduler,
                   dask_scheduler_port,
                   dask_workers,
                   out,
                   intermediate_results,
                   checkpoint_period,
                   os.path.abspath(pyenv))


def main():
    run_simulator()
    print("Simulation done.")

if __name__ == "__main__":
    main()
