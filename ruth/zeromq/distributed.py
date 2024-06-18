import click
import logging

import pandas as pd
import json
import os
from pathlib import Path

from ..zeromq.bench import get_slurm_nodes, run


@click.command()
@click.argument("experiment-name", type=str)
@click.argument("evkit-dir-path", type=click.Path(exists=True))
@click.option("--config-file", type=click.Path(exists=True), help="Path to simulation config.", default="config.json")
@click.option("--workers", type=int, default=32, help="Number of workers. Default 32.")
@click.option("--spawn-workers-at-main-node", is_flag=True, help="Spawn workers at main node.")
@click.option("--try-to-kill", is_flag=True, help="Try to kill workers after simulation is computed.")
def distributed(experiment_name, evkit_dir_path, config_file, workers, spawn_workers_at_main_node, try_to_kill):
    work_dir = Path(os.getcwd()).absolute()
    worker_dir = work_dir / experiment_name
    env_path = os.environ["VIRTUAL_ENV"]
    modules = [
        "Python/3.10.8-GCCcore-12.2.0",
        "GCC/12.2.0",
        "SQLite/3.39.4-GCCcore-12.2.0",
        "HDF5/1.14.0-gompi-2022b",
        "CMake/3.24.3-GCCcore-12.2.0",
        "Boost/1.81.0-GCC-12.2.0"
    ]
    hosts = get_slurm_nodes()

    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(module)s:%(levelname)s %(message)s")

    result = run(
        workers=workers,
        hosts=hosts,
        WORK_DIR=work_dir,
        WORKER_DIR=worker_dir,
        CONFIG_FILE=config_file,
        EVKIT_PATH=evkit_dir_path,
        MODULES=modules,
        ENV_PATH=env_path,
        try_to_kill=try_to_kill,
        spawn_workers_at_main_node=spawn_workers_at_main_node
    )

    # Save
    json_data = json.dumps(result)
    df = pd.read_json(json_data)
    df.to_csv(f"{worker_dir}/results.csv", index=False)


if __name__ == "__main__":
    distributed()
