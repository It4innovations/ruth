# Distribution
In order to run distribution following libraries are required in the envrionemnt:

- evkit   https://code.it4i.cz/everest/evkit
- cluster https://code.it4i.cz/def/cluster
- ruth https://github.com/It4innovations/ruth

Cluster is installed if instruction for evkit are followed (--recursive clone).

## Running workers
Consider config file as defined in ```config.json```:
```
{
  "ruth-simulator":
  {
    ...
    "k-alternatives": 1,
    ...
  },
  "run" :
  {
    "vehicles-path": "input.parquet"
  },
  "alternatives" : {
    "dijkstra-fastest": 0.0,
    "dijkstra-shortest": 0.0,
    "plateau-fastest": 1.0
  },
  "route-selection" : {
    "first": 1.0,
    "random": 0.0,
    "ptdr": 0.0
  },
  "distribution": {
    "number-of-workers": 1,
    "evkit-dir-path": "evkit",
    "spawn-workers-at-main-node": true,
    "try-to-kill": false
  }
}
```
Where we set the map and k-alternatives accordingly, together with plateau-fastest in order to use ZeroMQ for 
distributed spawning of workers across nodes.

In section ```distribution``` we set the number of workers, path to evkit and other parameters that are used for a distributed run.
Then we can run the simulation with:
```
ruth-distributed --config-file="config.json"
```

We can also directly use ```bench.py```, specifically function ```run```.
For correct incorporation of nodes spawned and configuration this function has to be used inside ```ruth```, otherwise 
we may simply use ```bench.py``` and edit it's parameters since it spawns the ```run``` with following parameters:
```
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
CONFIG_FILE = "config.json"
EVKIT_PATH = "evkit_dir_path"
hosts = get_slurm_nodes()
workers = 32
try_to_kill = False

run(...)
```

## Submitting a job
In order to submit a job to a cluster, for example SLURM, we may use:
```
#!/usr/bin/bash
#SBATCH --nodes 2

ml purge
ml Python/3.10.8-GCCcore-12.2.0
ml GCC/12.2.0
ml SQLite/3.39.4-GCCcore-12.2.0
ml HDF5/1.14.0-gompi-2022b
ml CMake/3.24.3-GCCcore-12.2.0
ml Boost/1.81.0-GCC-12.2.0

source VENV_PATH
ruth-distributed --config-file="config.json"
```

## Running benchmark
Benchmark is using run as a background process in order to execute all experiments.
Example of running a benchmark can be seen below:
```
WORK_DIR = Path(os.getcwd()).absolute()
WORKER_DIR = WORK_DIR / "bench_example"
ENV_PATH = os.environ["VIRTUAL_ENV"]
MODULES = [
    "Python/3.10.8-GCCcore-12.2.0",
    "GCC/12.2.0",
    "SQLite/3.39.4-GCCcore-12.2.0",
    "HDF5/1.14.0-gompi-2022b",
    "CMake/3.24.3-GCCcore-12.2.0",
    "Boost/1.81.0-GCC-12.2.0"
]
CONFIG_FILE = "config.json"
EVKIT_PATH = "evkit_path"
hosts = get_slurm_nodes()
workers = [1, 2]
try_to_kill = False
spawn_workers_at_main_node = False
repeats = 1

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(module)s:%(levelname)s %(message)s")

result = bench(
    workers=workers,
    hosts=hosts,
    OUTPUT_DIR=WORKER_DIR,
    CONFIG_FILE=CONFIG_FILE,
    EVKIT_PATH=EVKIT_PATH,
    MODULES=MODULES,
    ENV_PATH=ENV_PATH,
    try_to_kill=try_to_kill,
    spawn_workers_at_main_node=spawn_workers_at_main_node,
    repeats=1
)
```
