import click
import logging

import os
from pathlib import Path

from serde.json import from_json

from ..tools.simulator import run_inner
from ..tools.simulator_conf import Args, fill_args
from ..zeromq.bench import get_slurm_nodes, run, get_modules, get_cpu_count


@click.command()
@click.option("--config-file", type=click.Path(exists=True), help="Path to simulation config.",
              default="config.json")
def distributed(config_file):
    if os.path.isfile(config_file):
        with open(config_file, 'r') as f:
            config_data = f.read()
            args = from_json(Args, config_data)
    else:
        logging.error(f"Config file {config_file} does not exist.")
        return

    if args.distribution is None:
        logging.error("Distributed settings are missing. Running single node simulation.")
        args, path = fill_args(config_file)
        run_inner(args.common, path, args.alternatives_ratio, args.route_selection_ratio)
        return

    experiment_name = args.common.task_id if args.common.task_id else 'run'
    evkit_dir_path = args.distribution.evkit_dir_path
    workers = args.distribution.number_of_workers
    max_workers = get_cpu_count()
    if workers > max_workers:
        workers = max_workers

    spawn_workers_at_main_node = args.distribution.spawn_workers_at_main_node
    try_to_kill = args.distribution.try_to_kill

    work_dir = Path(os.getcwd()).absolute()
    worker_dir = work_dir / experiment_name
    env_path = os.environ["VIRTUAL_ENV"]
    modules = get_modules()
    hosts = get_slurm_nodes()

    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=f'{worker_dir}/bench.log', level=logging.DEBUG,
                        format="%(asctime)s %(module)s:%(levelname)s %(message)s")


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
        spawn_workers_at_main_node=spawn_workers_at_main_node,
        logger=logger
    )


if __name__ == "__main__":
    distributed()
