# ruth

<p align="center">
    <img width="60%" src="docs/images/ruth.svg?sanitize=true">
</p>

A python library for routing on OSM map based on [osmnx](https://github.com/gboeing/osmnx) and [probduration](https://code.it4i.cz/everest/py-probduration).

## Documentation

The detailed documentation is available at: https://it4innovations.github.io/ruth/

## Installation
Required python version >= 3.9.6
``` sh
# requirements
sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    git curl \
    build-essential gdal-bin libgdal-dev openmpi-bin libopenmpi-dev \
    python3 python3-dev python3-virtualenv python3-pip python3-setuptools python3-wheel

# install rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# create and activate python virtual environment
python3 -m venv venv
source venv/bin/activate

# within the venv
# install and update python dependencies
python3 -m pip install -U pip setuptools wheel cython

# install ruth - the traffic simulator
python3 -m pip install git+https://github.com/It4innovations/ruth.git
```

## Test Run

* activate virtual environment with `ruth` installed
* use the files in `benchmarks/od-matrix/`
* use configuration file to run the simulation
    ``` sh
    ruth-simulator-conf --config-file="config.json" run 
    ```
* configuration file with all options:
  ``` json
  {
    "ruth-simulator":
    {
      "departure-time" :"2021-08-3 00:00:00",
      "round-frequency-s" : 5,
      "k-alternatives": 1,
      "map-update-freq-s" : 1,
      "los-vehicles-tolerance-s" : 5,
      "out" : "simulation_record.pickle",
      "seed": 7,
      "walltime-s" : 2000,
      "saving-interval-s" : 100,
      "speeds-path" : "",
      "ptdr-path" : "",
      "continue-from": ""
    },
    "run" :
    {
      "vehicles-path": "benchmarks/od-matrices/INPUT-od-matrix-10-vehicles-town-resolution.parquet",
      "alternatives": "fastest-paths",
      "route-selection": "first"
    }
  }
  ```
* or use command line arguments
    ``` sh
    ruth-simulator --departure-time="2021-06-16 07:00:00" --k-alternatives=4 --out=simulation_record.pickle --seed=7 run INPUT-od-matrix-10-vehicles-town-resolution.parquet
    ```

### Options
#### Alternatives
- shortest-paths networkx implementation of dijkstra algorithm using route length as weight
- fastest-paths: networkx implementation of dijkstra algorithm using current-travel-time as weight
- distributed: cpp implementation of Plateau algorithm
#### Route selection
- first: uses the first from found alternatives
- random: uses random alternative
- distributed: uses alternative selected by PTDR (ptdr_path needed)
#### Speeds path
- path to csv file with temporary max speeds (example below)
    ``` csv
    node_from;node_to;speed;timestamp_from;timestamp_to
    8400868548;10703818;0;2021-08-3 00:10:00;2021-08-3 00:25:00
    ```
#### PTDR path
- path to msqpack file with probability speed profiles
- to generate PTDR file use first the `aggregate-globalview` or `aggregate-globalview-set` command generating
csv file with aggregated information about speeds during simulation
  ``` csv
  segment_osm_id;fcd_time_calc;segment_length;max_speed;current_speed
  OSM172512T300107261;2023-11-13 00:00;74.59;46.7;46.7
  OSM172512T300107261;2023-11-20 00:00;74.59;46.7;12.97
  ``` 
- the use [this code](https://code.it4i.cz/mic0427/ptdr/) to calculate probability profiles
  (json representation of msqpack file below)
  ``` json
  [
    { "id": "OSM172512T300107261",
      "length": 74.59,
      "max_speed": 46.7,
      "profiles": [{
          "time_id": 0,
          "values": [0, 0, 27, 100],
          "cumprobs": [0.0, 0.0, 0.5, 1.0]
         },
         ...
      ]
    },
    ...
  ]
  ```