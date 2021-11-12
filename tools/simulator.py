# TODO come up with better name

import os
import click
from datetime import datetime
from cluster import cluster


@click.command()
@click.argument("input_csv", type=click.Path(exists=True))
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
@click.option("--dask-workers", multiple=True, default=["localhost"]) # TODO: do I want to spawn more workers on the same address
@click.option("--out", type=str, default="out.pickle")
@click.option("--intermediate-results",
              type=click.Path(),
              help="A path to the folder with intermediate results.")
def main(input_csv,
         departure_time,
         k_routes,
         n_samples,
         seed,
         gv_update_period,
         dask_scheduler,
         dask_scheduler_port,
         dask_workers,
         out,
         intermediate_results):


    # TODO: move it to it4i version or **get rid of conda**
    conda_path = f"{os.getenv('CONDA_PREFIX')}/etc/profile.d/conda.sh"


    # pbs_nodefile = os.getenv("PBS_NODEFILE")
    # if intermediate_results is None:
    #     basename = os.path.basename(pbs_nodefile)
    #     os.mkdir(basename)
    #     intermediate_results = os.path.abspath(basename)

    # with open(pbs_nodefile) as f:
    #     hosts = f.readlines()
    #     nodes = [host.strip() for host in hosts]

    cluster.start_process(["dask-scheduler",
                           "--port",
                           dask_scheduler_port,
                           "--preload",
                           "/home/sur096/ruth/add-to-syspath.py"],
                          host=dask_scheduler,
                          init_cmd=[f"source {conda_path} && conda activate ox"])

    for worker in dask_workers:
        cluster.start_process(["dask-worker",
                               f"{dask_scheduler}:{dask_scheduler_port}",
                               "--preload",
                               "/home/sur096/ruth/add-to-syspath.py"],
                              host=worker,
                              init_cmd=[f"source {conda_path} && conda activate ox"])

    final_state_df = simulate(input_csv,
                              len(dask_workers),
                              departure_time,
                              k_routes,
                              n_samples,
                              seed,
                              gv_update_period,
                              dask_scheduler,
                              dask_scheduler_port,
                              intermediate_results)

    out_path = os.path.abspath(out)
    final_state_df.to_pickle(out_path)
    print("Simulation done.")


if __name__ == "__main__":
    main()
