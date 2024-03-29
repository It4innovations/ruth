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
    "vehicles-path": "/mnt/proj1/dd-23-154/bbox_parquets/250K_19022024.parquet"
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
  }
}
```

Where we set the map and k-alternatives accordingly, together with plateau-fastest in order to use ZeroMQ for 
distributed spawning of workers accross nodes.

In order to run the simulation in distributed fashion we can use ```bench.py```, specifically function ```run```.
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
CONFIG_FILE = "/mnt/proj2/open-27-41/simulator/ruth/config.json"
EVKIT_PATH = "/mnt/proj2/open-27-41/simulator/evkit"
hosts = get_slurm_nodes()
workers = 32
try_to_kill = False

run(...)
```
Or run the command:
```
ruth-distributed EXPERIMENT_NAME EVKIT_DIR_PATH --config-file="config.json" --workers=32
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
python3 bench.py EXPERIMENT_NAME
```