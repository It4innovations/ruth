import logging
import pandas as pd
import time
import json
import socket
import subprocess
import os
import sys
import dataclasses

from typing import List
from pathlib import Path
from contextlib import closing
from dataclasses import dataclass
from cluster.cluster import Cluster, start_process


def get_pbs_nodes() -> List[str]:
    with open(os.environ["PBS_NODEFILE"]) as f:
        return list(l.strip() for l in f)


def get_slurm_nodes() -> List[str]:
    output = subprocess.getoutput("scontrol show hostnames")
    return output.split('\n')


def find_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def is_port_open(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2)
    try:
        s.connect(("", port))
        return True
    except:
        return False


def is_running(pid):
    """ Check For the existence of a unix pid. """
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True


@dataclass
class RunResult:
    repeat: int
    output: str
    duration: float


def run(workers: int,
        hosts: List[str],
        WORK_DIR,
        WORKER_DIR,
        CONFIG_FILE: str,
        EVKIT_PATH: str,
        MODULES: List[str],
        ENV_PATH,
        try_to_kill: bool,
        spawn_workers_at_main_node: bool):
    """
    Run the workers in a distributed fashion by spawning them on multiple host(s), nodes.
    workers                     = amount of workers
    WORKER_DIR                  = where the workers will be stored
    CONFIG_FILE                 = path to config file (from ruth)
    EVKIT_PATH                  = path to evkit
    MODULES                     = modules used in order to spawn workers
    ENV_PATH                    = path to the environment
    try_to_kill                 = experimental, tries to kill workers after simulation is computed
    spawn_workers_at_main_node  = experimental, spawns workers at the same node where main process is located

    Potential Issue:
    -   Killing workers may result in error due to rights (connected to cluster library)
    """
    # Find open ports
    while True:
        port = find_free_port()
        management_port = find_free_port()
        if port != management_port:
            break

    # Set Client
    CLIENT_ADDRESS = hosts[0]
    print(f'Client is running at: {CLIENT_ADDRESS}')

    # Start main
    target_dir = Path(EVKIT_PATH)
    target_dir.mkdir(parents=True, exist_ok=True)
    build_process = start_process(
        commands=[f"cargo build --release --features alternatives,rpath --manifest-path {EVKIT_PATH}/Cargo.toml"],
        workdir=str(target_dir),
        env={"RUST_LOG": "debug"},
        pyenv=str(ENV_PATH),
        modules=MODULES,
        hostname=CLIENT_ADDRESS,
        name=f"target"
    )

    # Wait until the process end
    print("Building the worker")
    while is_running(build_process.pid):
        time.sleep(0.25)
    print("Worker built")

    # Include spawning of workers on same node
    if spawn_workers_at_main_node == True:
        starting_node = 0
    else:
        starting_node = 1

    try:
        cluster_data = Cluster(str(WORK_DIR))
        print(f"Saving to: {WORKER_DIR}")
        print(f"Environment in: {ENV_PATH}")
        print(f"Running client at: {CLIENT_ADDRESS}")
        output = WORKER_DIR / f"workers_{workers}"

        start_time = time.time()
        for n in range(starting_node, len(hosts)):
            HOST_ADDRESS = hosts[n]

            #start_time = time.time()
            for i in range(workers):
                print(f"Creating worker {i} at host {HOST_ADDRESS}")
                worker_dir = output / f"node_{n}" / f"worker_{i}"
                worker_dir.mkdir(parents=True, exist_ok=True)
                worker_process = start_process(
                    commands=[
                        f"{target_dir}/target/release/worker run {CLIENT_ADDRESS}:{port} {CLIENT_ADDRESS}:{management_port}"],
                    workdir=str(worker_dir),
                    env={"RUST_LOG": "debug"},
                    pyenv=str(ENV_PATH),
                    modules=MODULES,
                    hostname=HOST_ADDRESS,
                    name=f"worker_{i}"
                )
                #time.sleep(1)
                cluster_data.add(worker_process)

        # Wait for workers to be spawned => Inside Client
        time.sleep(60)
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Spawning of workers took: {total_time}")

        # Start main
        start = time.time()
        main_dir = output / f"main"
        main_dir.mkdir(parents=True, exist_ok=True)
        main_process = start_process(
            commands=[f"ruth-simulator-conf --config-file={CONFIG_FILE} run"],
            workdir=str(main_dir),
            pyenv=str(ENV_PATH),
            env={"port": port, "broadcast_port": management_port},
            modules=MODULES,
            hostname=CLIENT_ADDRESS,
            name=f"main"
        )
        cluster_data.add(main_process)

        # Start timer since we're running simulation
        start = time.time()
        print("Runing the main computation")
        while is_running(main_process.pid):
            time.sleep(0.25)
        end = time.time()
        duration = end - start
        print(f"Workers per node: {workers}, Nodes: {len(hosts) - 1}, computation time: {duration}")
        if try_to_kill == True:
            cluster_data.kill()

        # Save to file and return info about run
        result = RunResult(repeat=0, output=output, duration=duration)
        json_data = json.dumps(dataclasses.asdict(result))
        df = pd.read_json(json_data)
        df.to_csv(f"{WORKER_DIR}/results.csv", index=False)
        return result

    except Exception as e:
        print(e)


def bench(workers: List[int],
          hosts: List[str],
          OUTPUT_DIR,
          CONFIG_FILE: str,
          EVKIT_PATH: str,
          MODULES: List[str],
          ENV_PATH,
          try_to_kill: bool,
          spawn_workers_at_main_node: bool,
          repeats: int) -> RunResult:
    """
    Run the benchmark on multiple host(s), nodes.
    workers                     = amount of workers
    WORKER_DIR                  = where the workers will be stored
    CONFIG_FILE                 = path to config file (from ruth)
    EVKIT_PATH                  = path to evkit
    MODULES                     = modules used in order to spawn workers
    ENV_PATH                    = path to the environment
    try_to_kill                 = experimental, tries to kill workers after simulation is computed
    spawn_workers_at_main_node  = experimental, spawns workers at the same node where main process is located
    repeats                     = number of amounts the experiment is repeated

    Potential Issue:
    -   Building the workers inside run, build them inside bench and then just run them
    """

    CLIENT_ADDRESS = hosts[0]
    print(f'Client is running at: {CLIENT_ADDRESS}')
    results = []

    # Preprocess workers based on if we try to kill them
    if try_to_kill is False:
        active_workers = workers[0]  # 1
        updated_workers = [workers[0]]  # [1]
        for idx in range(1, len(workers)):
            w = workers[idx]
            new_workers = w - active_workers
            updated_workers.append(new_workers)
            active_workers += new_workers
        workers = updated_workers

    # Main loop for benchmark
    for r in range(repeats):
        port = find_free_port()
        management_port = port + 1

        for w in workers:
            WORK_DIR = OUTPUT_DIR / str(r)
            WORKER_DIR = OUTPUT_DIR / str(r) / str(w)
            try:
                result = run(w, hosts, WORK_DIR, WORKER_DIR, CONFIG_FILE, EVKIT_PATH, MODULES, ENV_PATH, try_to_kill,
                             spawn_workers_at_main_node)
                results.append(result)
                time.sleep(10)

            except KeyboardInterrupt:
                continue

    # Save
    json_data = json.dumps(dataclasses.asdict(result))
    df = pd.read_json(json_data)
    df.to_csv(f"{OUTPUT_DIR}/results.csv", index=False)

    return results


if __name__ == "__main__":
    WORK_DIR = Path(os.getcwd()).absolute()
    WORKER_DIR = WORK_DIR / str(sys.argv[1])
    ENV_PATH = os.environ["VIRTUAL_ENV"]
    MODULES = [
        "Python/3.10.8-GCCcore-12.2.0",
        "GCC/12.2.0",
        "SQLite/3.39.4-GCCcore-12.2.0",
        "HDF5/1.14.0-gompi-2022b",
        "CMake/3.24.3-GCCcore-12.2.0",
        "Boost/1.81.0-GCC-12.2.0"
    ]
    # CHANGE PATHS
    CONFIG_FILE = ""
    EVKIT_PATH = ""
    hosts = get_slurm_nodes()
    workers = 32
    try_to_kill = False
    spawn_workers_at_main_node = False

    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(module)s:%(levelname)s %(message)s")

    result = run(
        workers=workers,
        hosts=hosts,
        WORK_DIR=WORK_DIR,
        WORKER_DIR=WORKER_DIR,
        CONFIG_FILE=CONFIG_FILE,
        EVKIT_PATH=EVKIT_PATH,
        MODULES=MODULES,
        ENV_PATH=ENV_PATH,
        try_to_kill=try_to_kill,
        spawn_workers_at_main_node=spawn_workers_at_main_node
    )

    # Save
    json_data = json.dumps(result)
    df = pd.read_json(json_data)
    df.to_csv(f"{WORKER_DIR}/results.csv", index=False)
