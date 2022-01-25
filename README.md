# ruth

A python library for routing on OSM map based on [osmnx](https://github.com/gboeing/osmnx) and [probduration](https://code.it4i.cz/everest/py-probduration).

To demonstrate the library functionality a single-node traffic simulator is implemented.

## Installation


``` sh
virtualenv venv
source venv/bin/activate
python3 -m pip install .
```

This install the simulator `ruth-simulator` and data preprocessing tool `ruth-data-preprocessing`.

## Running
The input datasets are available in `benchmarks` folder. The `ruth-data-preprocessing` prepares the data for the simulator.

The simulator contains two versions:
1) generic - allow user to specify the _dask-_ 
2) pbs - set up dask-scheduler and workers accoriding to PDBS file. The first node is used the dask-scheduler and the rest as workers.

The simulator have other options to set up. See more `ruth-simulator generic --help`.

### Example:

``` sh
ruth-data-preprocessing ./benchmarks/antarex/INPUT-cars_10.csv --out INPUT-cars_10.pickle
ruth-simulator generic INPUT-cars_10.pickle --departure-time "2022-01-24 18:04:00" --k-routes 4 --n-samples 10 --gv-update-period 10 --out res.pickle --pyenv $(pwd)/venv/
```

### Limitations/known issues

 * Currently _no limit probability profile_ is used; therefore the delay will always be zero and the simulation degrades into shortest path routing. Nevertheless, all the steps and computentional demanding parts are present.
## Usage
TODO:

### Downloading map layer
TODO: explain how to the maps are downloaded and how to extend the supported areas

### Routing
TODO: explain hierarchical routing; not implemented yet.

