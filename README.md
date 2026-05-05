# RUTH 

RUTH is a traffic simulator for large-scale routing, traffic simulation, and mobility data generation using OpenStreetMap road networks. The main simulation workflow is exposed through Python, while selected performance-critical components are implemented in C++. Ongoing development continues to extend the C++ components to improve execution efficiency for larger workloads and HPC environments.

<p align="center">
    <img width="60%" src="docs/images/ruth.svg?sanitize=true">
</p>

## Installation

### Basic Installation

Python packages listed in `requirements.txt` are pinned for Python 3.9 - 3.11. Change versions in `requirements.txt` if needed.

Create and activate a Python virtual environment:

```bash
python3.11 -m venv venv
source venv/bin/activate
```

Then install `ruth` using one of the following options.

#### Option 1: Install from a local copy of this branch

```bash
git clone --recurse-submodules --branch v2.4-globalview-optimization https://github.com/It4innovations/ruth.git
cd ruth
pip install -e .
```

#### Option 2: Install this branch directly from GitHub

```bash
pip install git+https://github.com/It4innovations/ruth.git@v2.4-globalview-optimization
```

#### Option 3: Offline installation, e.g. HPC without Internet

Use this option when the target machine or cluster does not have Internet access.

On a machine with Internet access, go to a local copy of the RUTH repository and prepare the local wheel directory:

```bash
python -m pip download -r requirements.txt -d ruth-wheels
```

Copy both the local RUTH repository and the `ruth-wheels/` directory to the cluster.

Then, on the cluster, from inside the RUTH repository, install using the local wheel directory:

```bash
python -m pip install --no-index --find-links ruth-wheels -e .
```



To use **Plateau** algorithm or **multi-node** execution, ACE library must be compiled from source.

#### Load Required Modules (HPC environments)

The exact modules depend on the target HPC system. Some systems using EasyBuild require loading a base module environment first to expose the software stack.

```bash
# Example, system-dependent:
# ml EB/apps EB/install
# or
# ml GCCcore/<version>
# or
# ml foss/<version>

ml Python/3.11.3-GCCcore-12.3.0 CMake/3.26.3-GCCcore-12.3.0 \
   HDF5/1.14.0-gompi-2023a OpenMPI/4.1.5-GCC-12.3.0 SQLite/3.42.0-GCCcore-12.3.0

# If there is no Internet connectivity on the machine, load pybind11 locally
# to avoid fetching it during the build.
ml pybind11/2.11.1-GCCcore-12.3.0
```

#### Build Instructions

```bash
cd ruth/binding
mkdir build && cd build
PYTHON_FOR_CMAKE=$(which python)

cmake .. \
  -DCMAKE_C_COMPILER=gcc \
  -DCMAKE_CXX_COMPILER=g++ \
  -DPYTHON_EXECUTABLE="$PYTHON_FOR_CMAKE"

make -j

# Add build directory to Python path
export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTHONPATH=$(pwd)/lib:$PYTHONPATH

# Print to verify
echo $PYTHONPATH
```

## Execution

1. **Prepare your environment:**
   ```bash
   source venv/bin/activate  # Activate virtual environment
   ```

2. **Get sample data:**
   - Use the example files in `benchmarks/od-matrices/`
   - Or download larger datasets from [Input Datasets](#input-datasets)

3. **Run a basic simulation:**
   ```bash
   ruth-simulator-conf --config-file="config.json" run
   ```

4. **View results:**
   - Output is saved in fcd_history-partXXXX.h5 file and specified pickle file
   - Use the animation tools to visualize traffic flow (see [Animation](#animation))

### Multi-node Execution with MPI

Each MPI process spawns threads to utilize all available cores on the node.


```bash
# Basic multi-node execution
mpirun -n 2 ruth-simulator-conf --config-file="config.json" run

# Example: 16 processes, 8 per node, 16 cores each, bind to core
mpirun -n 16 --map-by ppr:8:node:PE=16 -bind-to core ruth-simulator-conf --config-file="config.json" run
```

## Tested Platforms

RUTH has been tested on local development environments and on several HPC systems. Local execution has been tested on Linux/x86_64 and macOS systems, including both Intel and Apple Silicon machines. HPC executions have been tested on Linux-based systems including Karolina, Barbora, LUMI-C, MareNostrum 5, and Deucalion, covering both x86_64 and ARM-based environments.

Platform-specific compiler, MPI, and module configurations may vary between systems. The build and run instructions may therefore require adaptation to the target environment.

### Configuration File

Configuration file with all available options:

```json
{
  "ruth-simulator": {
    "departure-time": "2026-05-05 07:00:00",
    "round-frequency-s": 5,
    "k-alternatives": 1,
    "map-update-freq-s": 1,
    "los-vehicles-tolerance-s": 5,
    "out": "simulation_record.pickle",
    "seed": 7,
    "walltime-s": 2000,
    "saving-interval-s": 100,
    "speeds-path": "",
    "travel-time-limit-perc": 0.1,
    "ptdr-path": "",
    "continue-from": "",
    "stuck-detection": 0,
    "plateau-default-route": false,
    "buffer-size": 10000,
    "max-records-per-file": 1000000000
  },
  "run": {
    "vehicles-path": "benchmarks/od-matrices/INPUT-od-matrix-10-vehicles.parquet"
  },
  "alternatives": {
    "dijkstra-fastest": 0.3,
    "dijkstra-shortest": 0.0,
    "plateau-fastest": 0.0
  },
  "route-selection": {
    "first": 0.3,
    "random": 0.0,
    "ptdr": 0.0
  }
}
```

**Configuration Parameters Explained:**

| Parameter                  | Description                                                  | Default                  |
|----------------------------|--------------------------------------------------------------|--------------------------|
| `departure-time`           | Simulation start time                                        | required                 |
| `round-frequency-s`        | Time step duration in seconds                                | 5                        |
| `k-alternatives`           | Number of alternative routes to compute                      | 1                        |
| `map-update-freq-s`        | Frequency to update traffic conditions                       | 1                        |
| `los-vehicles-tolerance-s` | Tolerance for vehicle loss-of-service detection              | 5                        |
| `out`                      | Output file path for simulation results                      | simulation-record.pickle |
| `seed`                     | Random seed for reproducibility                              | random                   |
| `walltime-s`               | Time limit when simulation is saved                          | -                        |
| `saving-interval-s`        | Interval for saving intermediate results                     | -                        |
| `speeds-path`              | Path to CSV with temporary speed restrictions                | -                        |
| `travel-time-limit-perc`   | Travel time limit as percentage                              | 0.1                      |
| `ptdr-path`                | Path to PTDR route selection data                            | -                        |
| `continue-from`            | Path to pickle file to continue previous simulation          | -                        |
| `stuck-detection`          | Number of steps before stuck detection triggers (0=disabled) | 0                        |
| `plateau-default-route`    | Recalculate default route with Plateau                       | false                    |
| `buffer-size`              | Number of FCD records to buffer before flushing to disk       | 10000                    |
| `max-records-per-file`     | Rotate HDF5 file after this many records                     | 1000000000               |

### Command-line Arguments

Alternatively, use command-line arguments instead of a configuration file:

```sh
ruth-simulator \
  --departure-time="2026-05-05 07:00:00" --k-alternatives=4 --seed=7 \
  run \
  --alt-dijkstra-fastest=0.3 --selection-random=0.3 \
  "INPUT-od-matrix-10-vehicles.parquet"
```

## Input Datasets

Pre-configured datasets are available for different simulation scales:

- 10K - 300K vehicles: [zenodo.13285293](https://doi.org/10.5281/zenodo.13285293)
- 2.6M - 25M vehicles: [zenodo.17206523](https://doi.org/10.5281/zenodo.17206523)

Sample data for testing is available in `benchmarks/od-matrices/`. 

## Configuration Options

### Alternative Route Generation

Configure how alternative routes are computed. Vehicles not assigned to an alternative method will use their default route from the input file.

**Available Methods:**

- **`dijkstra-shortest`**: NetworkX Dijkstra using route length as weight
- **`dijkstra-fastest`**: NetworkX Dijkstra using current travel time as weight  
- **`plateau-fastest`**: C++ implementation of Plateau algorithm (requires [ACE library](https://opencode.it4i.eu/epicure/ace))

  - Set `plateau-default-route: true` to recalculate default routes with Plateau

### Route Selection Strategy

Configure how vehicles select from computed alternative routes:

- **`first`**: Always use the first alternative found
- **`random`**: Randomly select from available alternatives
- **`ptdr`**: Use PTDR algorithm (requires `ptdr-path` in configuration)

> **Note:** Sum of alternative percentages must equal sum of selection percentages.

### Speed Restrictions

You can specify temporary speed restrictions via CSV file using the `speeds-path` parameter:

```csv
node_from;node_to;speed;timestamp_from;timestamp_to
8400868548;10703818;0;2026-08-03 00:10:00;2026-08-03 00:25:00
```

## Animation
To create animation of the simulation, `FFmpeg` needs to be installed. Using option `--gif` to generate 
gif instead of mp4 does not require `FFmpeg`.

There are two types of animation:
- **Traffic volume animation** - Both color and width of route segments are based on the number of vehicles.
- **Traffic speed animation** - Color is based on the current speed on the segment. Width is based on the number of vehicles.

For more information how to create an animation of an existing simulation, see **ruth/flowmap/README.md**.


## Tools
Other tools can be found in the `ruth/tools` directory. See **ruth/tools/README.md**.


# Acknowledgement

RUTH has been developed and extended through several research activities at IT4Innovations. Contributions to its development, optimisation, deployment, and large-scale execution experiments have been supported by projects and initiatives including LEXIS,EVEREST, EuroCC, EuroCC2, EuroHPC compute-time projects, and INNOVAITE.
