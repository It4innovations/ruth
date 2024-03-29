import itertools
import logging
import pandas as pd
import time
import json
import socket
import subprocess
import os
import sys
import signal

from typing import List
from pathlib import Path
from collections import defaultdict
from contextlib import closing
from dataclasses import dataclass
from cluster.cluster import Cluster, start_process
from cluster import cluster
from src.client import Client


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
        WORKER_DIR,
        CONFIG_FILE: str,
        EVKIT_PATH: str,
        MODULES: List[str],
        ENV_PATH,
        try_to_kill:bool,
        spawn_workers_at_main_node:bool):
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

    Problem:
    Killing workers may result in error due to rights (connected to cluster library)

    Example Usage:
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
    CONFIG_FILE = "/mnt/proj2/open-27-41/simulator/ruth/config.json"
    EVKIT_PATH = "/mnt/proj2/open-27-41/simulator/evkit"
    hosts = get_slurm_nodes()
    """
    # Find open port
    port = find_free_port()
    management_port = port + 1

    # Set Client
    CLIENT_ADDRESS = hosts[0]
    print(f'Client is running at: {CLIENT_ADDRESS}')

    # Start main
    target_dir = Path(EVKIT_PATH)
    target_dir.mkdir(parents=True, exist_ok=True)
    build_process = start_process(
        commands=[f"cargo build --release --features alternatives,rpath --manifest-path {EVKIT_PATH}/Cargo.toml"],
        workdir=str(target_dir),
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

        for n in range(starting_node, len(hosts)):
            HOST_ADDRESS = hosts[n]

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
                time.sleep(1)
                cluster_data.add(worker_process)

        # Wait for workers to be spawned
        time.sleep(30)

        # Start main
        main_dir = output / f"main"
        main_dir.mkdir(parents=True, exist_ok=True)
        main_process = start_process(
            commands=[f"ruth-simulator-conf --config-file={CONFIG_FILE} run"],
            workdir=str(main_dir),
            pyenv=str(ENV_PATH),
            env={"port": port},
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
        return RunResult(repeat=0, output=output, duration=duration)

    except Exception as e:
        print(e)


def bench(hosts: List[str], WORKER_DIR) -> RunResult:
    CLIENT_ADDRESS = hosts[0]
    print(f'Client is running at: {CLIENT_ADDRESS}')
    CONFIG_FILE = "/mnt/proj2/open-27-41/simulator/ruth/config.json"

    # Start main
    target_dir = Path("/mnt/proj2/open-27-41/simulator/evkit")
    target_dir.mkdir(parents=True, exist_ok=True)
    target_process = start_process(
        commands=[
            f"cargo build --release --features alternatives,rpath --manifest-path /mnt/proj2/open-27-41/simulator/evkit/Cargo.toml"],
        workdir=str(target_dir),
        pyenv=str(ENV_PATH),
        modules=MODULES,
        hostname=CLIENT_ADDRESS,
        name=f"target"
    )

    # Wait until the process end
    print("Building the worker")
    while is_running(target_process.pid):
        time.sleep(0.25)
    print("Worker built")

    results = []
    repeats = 3
    workers = [32]  # [1, 1, 2, 4, 8, 16, 32] #[1, 2, 4, 8, 16, 32, 64]
    for r in range(repeats):
        total = 0

        # Find two open ports
        port = find_free_port()
        management_port = port + 1

        for w in workers:
            total += w
            try:
                cluster_data = Cluster(str(WORK_DIR))
                print(f"Saving to: {WORKER_DIR}")
                print(f"Environment in: {ENV_PATH}")
                print(f"Running client at: {CLIENT_ADDRESS}")
                output = WORKER_DIR / f"workers_{total}" / f"repeat_{r}"

                for n in range(1, len(hosts)):
                    HOST_ADDRESS = hosts[n]

                    for i in range(w):
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
                        time.sleep(1)
                        # cluster_data.add(worker_process)

                # Wait for workers to be spawned
                time.sleep(60)

                # Start main
                main_dir = output / f"main"
                main_dir.mkdir(parents=True, exist_ok=True)
                main_process = start_process(
                    commands=[f"ruth-simulator-conf --config-file={CONFIG_FILE} run"],
                    workdir=str(main_dir),
                    pyenv=str(ENV_PATH),
                    env={"port": port},
                    modules=MODULES,
                    hostname=CLIENT_ADDRESS,
                    name=f"main"
                )
                cluster_data.add(main_process)

                # Start
                start = time.time()

                # Wait until the process end
                print("Runing the main computation")
                while is_running(main_process.pid):
                    time.sleep(0.25)
                print("Main computation ended#")

                end = time.time()
                duration = end - start
                print(f"Workers per node: {w}, Nodes: {len(hosts) - 1}, computation time: {duration}")

                results.append(RunResult(repeat=r, output=output, duration=duration))
                time.sleep(10)

            except KeyboardInterrupt:
                continue

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
    CONFIG_FILE = "/mnt/proj2/open-27-41/simulator/ruth/config.json"
    EVKIT_PATH = "/mnt/proj2/open-27-41/simulator/evkit"
    hosts = get_slurm_nodes()
    workers = 32
    try_to_kill = False
    spawn_workers_at_main_node = False

    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(module)s:%(levelname)s %(message)s")

    result = run(
        workers=workers,
        hosts=hosts,
        WORKER_DIR=WORKER_DIR,
        CONFIG_FILE=CONFIG_FILE,
        EVKIT_PATH=EVKIT_PATH,
        MODULES=MODULES,
        ENV_PATH=ENV_PATH,
        try_to_kill=try_to_kill,
        spawn_workers_at_main_node=spawn_workers_at_main_node
    )
    # result = bench(nodes, WORKER_DIR)

    # Save
    json_data = json.dumps(result)
    df = pd.read_json(json_data)
    df.to_csv(f"{WORKER_DIR}/results.csv", index=False)
