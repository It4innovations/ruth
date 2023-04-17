import itertools
import logging
import pandas as pd
import time
import json
import socket
import subprocess
import os

from typing import List
from pathlib import Path
from collections import defaultdict
from contextlib import closing
from dataclasses import dataclass
from cluster.cluster import Cluster, start_process
from cluster import cluster
from src.client import Client


WORK_DIR = Path(os.getcwd()).absolute()
# Virtual env is taken from whoever is running this script
ENV_PATH = os.environ["VIRTUAL_ENV"]
PYTHON_MODULE = "Python/3.8.6-GCCcore-10.2.0"

WORKER_DIR = WORK_DIR / "workers"


def kill_process(hostname: str, pid: int, signal="TERM"):
    """
    Kill a process with the given `pid` on the specified `hostname`
    :param hostname: Hostname where the process is located.
    :param pid: PGID of the process to kill.
    :param signal: Signal used to kill the process. One of "TERM", "KILL" or "INT".
    """
    import signal as pysignal

    assert signal in ("TERM", "KILL", "INT")
    cluster.logging.debug(f"Killing PGID {pid} on {hostname}")
    if not cluster.is_local(hostname):
        res = subprocess.run([f'ssh {hostname} kill -{signal} {pid}'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        if res.returncode != 0:
            cluster.logging.error(
                f"error: {res}")
            return False
    else:
        if signal == "TERM":
            signal = pysignal.SIGTERM
        elif signal == "KILL":
            signal = pysignal.SIGKILL
        elif signal == "INT":
            signal = pysignal.SIGINT
        os.killpg(pid, signal)
    return True


def get_pbs_nodes() -> List[str]:
    with open(os.environ["PBS_NODEFILE"]) as f:
        return list(l.strip() for l in f)


@dataclass
class RunResult:
    duration: float
    op_per_sec: float
    optimal: float


def find_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def run(map, hosts: List[str], workers_per_node: int) -> RunResult:
    port = find_free_port()
    client = Client(port=port)

    CLIENT_ADDRESS = hosts[0]
    cluster_data = Cluster(str(WORK_DIR))

    for i, host in enumerate(hosts[1:]):
        for i in range(workers_per_node):
            worker_dir = WORKER_DIR / f"worker_{i}"
            worker_dir.mkdir(parents=True, exist_ok=True)
            process = start_process(
                commands=[f"python3 {WORK_DIR}/ex_worker.py --address {CLIENT_ADDRESS} --port {port} --map {map}"],
                workdir=str(worker_dir),
                pyenv=str(ENV_PATH),
                modules=[PYTHON_MODULE],
                hostname=host,
                name=f"worker_{i}"
            )
            cluster_data.add(process)

    # Wait for workers to be spawned
    time.sleep(10)

    array = [[x, x + 1] for x in range(msg_count)]
    array = [json.dumps(x).encode() for x in array]

    start = time.time()
    results = client.compute(array)
    end = time.time()

    duration = end - start
    print(f'Computation time: {duration} for messages: {msg_count}')
    print(f'Cars per second: {msg_count/(duration)}')

    cluster_data.kill()
    return RunResult(duration=duration, op_per_sec=msg_count/duration, optimal=(sleep_time*msg_count)/(workers_per_node*(len(hosts)-1)))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s %(module)s:%(levelname)s %(message)s")

    pbs_nodes = get_pbs_nodes()

    sleep_times = [0.001, 0.1, 1]
    msg_counts = [100, 500, 1000]
    workers_per_nodes = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    data = defaultdict(list)
    for (sleep_time, msg_count, workers_per_node) in itertools.product(sleep_times, msg_counts, workers_per_nodes):
        print(f"Sleep: {sleep_time}, Workers: {workers_per_node}, Messages: {msg_count}")
        result = run(pbs_nodes, sleep_time, msg_count, workers_per_node)
        data["program-runtime"].append(sleep_time)
        data["workers"].append(workers_per_node)
        data["msg-count"].append(msg_count)
        data["duration"].append(result.duration)
        data["op-per-sec"].append(result.op_per_sec)
        data["optimal"].append(result.optimal)

    df = pd.DataFrame(data)
    df.to_csv("results.csv", index=False)
