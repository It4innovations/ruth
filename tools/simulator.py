import os
import click
from datetime import datetime

from ruth.distsim import simulate
from ruth.everestsim import simulate as everest_simulate

def dask_simulator(input_benchmark_data,
                   departure_time,
                   k_routes,
                   n_samples,
                   seed,
                   gv_update_period,
                   dask_scheduler,
                   dask_scheduler_port,
                   out,
                   intermediate_results,
                   checkpoint_period):
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
    out: Path
        A path of output file.
    intermediate_results: Optional[Path]
        A path of directory where the intermediate results are stored. If `None` the intermediate
        results are not stored.
    checkpoint_period: int
        A period in which the intermediate results are stored.
    """

    (gv, final_state_df) = simulate(input_benchmark_data,
                                    dask_scheduler,
                                    dask_scheduler_port,
                                    departure_time,
                                    k_routes,
                                    n_samples,
                                    seed,
                                    gv_update_period,
                                    intermediate_results,
                                    checkpoint_period)

    out_path = os.path.abspath(out)
    # TODO: solve the storing of resutl
    # final_state_df.to_pickle(out_path)
    gv.store(out_path)

    for v in final_state_df:
        print (v)


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
@click.option("--out", type=str, default="out.pickle")
@click.option("--intermediate-results",
              type=click.Path(),
              help="A path to the folder with intermediate results.")
@click.option("--checkpoint-period",
              type=int,
              default=1,
              help="A period in which the intermediate results are stored.")
def generic(input_benchmark_data,
            departure_time,
            k_routes,
            n_samples,
            seed,
            gv_update_period,
            dask_scheduler,
            dask_scheduler_port,
            out,
            intermediate_results,
            checkpoint_period):
    """Generic command calling the simulator within dask environment."""

    dask_simulator(input_benchmark_data,
                   departure_time,
                   k_routes,
                   n_samples,
                   seed,
                   gv_update_period,
                   dask_scheduler,
                   dask_scheduler_port,
                   out,
                   intermediate_results,
                   checkpoint_period)


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
@click.option("--out", type=str, default="out.pickle")
@click.option("--intermediate-results",
              type=click.Path(),
              help="A path to the folder with intermediate results.")
@click.option("--checkpoint-period",
              type=int,
              default=1,
              help="A period in which the intermediate results are stored.")
def everest(input_benchmark_data,
            departure_time,
            k_routes,
            n_samples,
            seed,
            gv_update_period,
            out,
            intermediate_results,
            checkpoint_period):
    """Generic command calling the simulator within dask environment."""

    final_state_df = everest_simulate(input_benchmark_data,
                                      departure_time,
                                      k_routes,
                                      n_samples,
                                      seed,
                                      gv_update_period,
                                      intermediate_results,
                                      checkpoint_period)

    #out_path = os.path.abspath(out)
    #final_state_df.to_pickle(out_path)



def main():
    run_simulator()
    print("Simulation done.")

if __name__ == "__main__":
    main()
