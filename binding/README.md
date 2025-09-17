## Modules

```bash
ml HDF5/1.14.0-gompi-2023a \
   SQLite/3.42.0-GCCcore-12.3.0 \
   CMake/3.26.3-GCCcore-12.3.0 \
   Python/3.11.3-GCCcore-12.3.0
```

## Build

```bash
cd binding
mkdir build
cd build
cmake ../
make
```

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

## Run Simulation with MPI

```bash
mpirun -np 2 ruth-simulator-conf --config-file="config.json" run
```