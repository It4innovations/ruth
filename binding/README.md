## Modules

```bash
ml Python/3.11.3-GCCcore-12.3.0
ml CMake/3.26.3-GCCcore-12.3.0
ml HDF5/1.14.0-gompi-2023a
ml OpenMPI/4.1.5-GCC-12.3.0
ml SQLite/3.42.0-GCCcore-12.3.0
```

## Build

```bash
cd ruth/binding
mkdir build
cd build
PYTHON_FOR_CMAKE=$(which python)
cmake .. -DPYTHON_EXECUTABLE="$PYTHON_FOR_CMAKE"
make
```

## Run Simulation with MPI
In this version each MPI process spawns threads to utilize all available cores on the node.
```bash
NPROCESSES=2
mpirun -n ${NPROCESSES} ruth-simulator-conf --config-file="config.json" run
```
- input dataset for 2.6M and 25M for multi-node execution can be found in this [dataset](https://doi.org/10.5281/zenodo.17206523)


## Test C++ Executable

Run the simulation with MPI:

```bash
mpirun -n 16 bash -c './build/ruth_exec --num-vehicles 100000'
```

Or with node/core mapping:

```bash
mpirun -n 16 --map-by ppr:8:node:PE=16 -bind-to core bash -c './build/ruth_exec --num-vehicles 100000'
```

## Test Python Bindings
Set Python path and run MPI test:

```bash
cd binding
export PYTHONPATH=path_to/binding/build:$PYTHONPATH
mpirun -np 2 python test.py
```