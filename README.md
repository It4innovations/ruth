# ruth

A python library for routing on OSM map based on [osmnx](https://github.com/gboeing/osmnx) and [probduration](https://code.it4i.cz/everest/py-probduration).

To demonstrate the library functionality a single-node traffic simulator is implemented.

## Usage
TODO:

### Downloading map layer
TODO: explain how to the maps are downloaded and how to extend the supported areas

### Routing
TODO: explain hierarchical routing; not implemented yet.

## Installation

### prerequisites

* redis


* install OSMnx follwing the instruction at: https://osmnx.readthedocs.io/en/stable/
* install pip within *ox* environment.

``` sh
conda install -n ox pip
```

* activate the conda environment and install *requirements.txt*

``` sh
conda activate ox
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r requirements.txt
```

## Single-node traffic simulator
The single-node traffic simulator performs the whole pipeline for simulating cars including monte carlo simulation for computing probability of delay on a route at a departure time (PTDR).

``` sh
python3 single-node-traffic-sim.py ./benchmarks/antarex/INPUT-cars_2.csv
```

The simulator have other options to set up. See more `python3 single-node-traffic-sim.py --help`.

### Limitations/known issues

 * Current version does not post-process the simulated traffic stored in redis yet.
 * Currently _no limit probability profile_ is used; therefore the delay will always be zero and the simulation degrades into shortest path routing. Nevertheless, all the steps and computentional demanding parts are present.
