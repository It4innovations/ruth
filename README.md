# ruth

<p align="center">
    <img width="60%" src="docs/images/ruth.svg?sanitize=true">
</p>

A python library for routing on OSM map based on [osmnx](https://github.com/gboeing/osmnx).

## Installation
Python packages listed in `requirements.txt` are pinned for Python 3.9 - 3.11.
Change versions in requirements.txt if needed.

Create and activate a python virtual environment and install `ruth`:
``` sh
python3 -m venv venv
source venv/bin/activate

# install ruth from GitHub
pip install git+https://github.com/It4innovations/ruth.git

# or install ruth from local copy
git clone --recurse-submodules https://github.com/It4innovations/ruth.git
pip install path_to/ruth
```
To use Plateau algorithm and **multi-node** execution, ACE has to be installed and running.
For more information, see **binding/README.md**.

## Execution

Information about **multi-node** execution can be found in **binding/README.md**.

For single-node execution, consider the following steps:

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
      "departure-time" :"2024-10-03 00:00:00",
      "round-frequency-s" : 5,
      "k-alternatives": 1,
      "map-update-freq-s" : 1,
      "los-vehicles-tolerance-s" : 5,
      "out" : "simulation_record.pickle",
      "seed": 7,
      "walltime-s" : 2000,
      "saving-interval-s" : 100,
      "speeds-path" : "",
      "travel-time-limit-perc": 0.1,
      "ptdr-path" : "",
      "continue-from": "",
      "stuck-detection": 0,
      "plateau-default-route": false
    },
    "run" :
    {
      "vehicles-path": "benchmarks/od-matrices/INPUT-od-matrix-10-vehicles.parquet"
    },
    "alternatives" : 
    {
      "dijkstra-fastest": 0.3,
      "dijkstra-shortest": 0.0,
      "plateau-fastest": 0.0
    },
    "route-selection" : {
      "first": 0.3,
      "random": 0.0,
      "ptdr": 0.0
    }
  }
  ```
* or use command line arguments
    ``` sh
    ruth-simulator --departure-time="2024-10-03 07:00:00" --k-alternatives=4 --out=simulation_record.pickle --seed=7 run --alt-dijkstra-fastest=0.3 --alt-plateau-fastest=0.0 --selection-first=0.3 --selection-random=0.0 --selection-ptdr=0.0 "INPUT-od-matrix-10-vehicles.parquet"
    ```
* input data for 10K-300K vehicle simulations can be found in this [dataset](https://doi.org/10.5281/zenodo.13285293).


## Options
For the Alternatives and Route selection, percentages of vehicles can be set for each type. The sum of percentages for Alternatives has to be equal to the sum for Route Selection. If the percentages don't add up to 1, no alternatives are calculated for the remaining vehicles and they stick to their original route that is in the input parquet file.
### Alternatives
- shortest-paths networkx implementation of dijkstra algorithm using route length as weight
- fastest-paths: networkx implementation of dijkstra algorithm using current-travel-time as weight
- distributed: cpp implementation of Plateau algorithm
  - provided by [ACE](https://opencode.it4i.eu/epicure/ace)
  - ports either taken from the environment or set to 5555 and 5556.
  - **plateau-default-route**: flag to recalculate the default route with Plateau algorithm
### Route selection
- first: uses the first from found alternatives
- random: uses random alternative
- distributed: uses alternative selected by PTDR (ptdr_path needed)
### Speeds path
- path to csv file with temporary max speeds (example below)
    ``` csv
    node_from;node_to;speed;timestamp_from;timestamp_to
    8400868548;10703818;0;2024-08-03 00:10:00;2024-08-03 00:25:00
    ```

### Stuck detection
Parameter `stuck-detection` defines the number of round-frequency-s long rounds with no vehicles movement
that have to pass before the simulation is terminated.  
The detection is based on all vehicles being in the same position for the whole time no temporary max speeds are to be updated.
If the parameter is **set to 0**, the detection is **disabled**.

## Tools
Other tools can be found in the `ruth/tools` directory. 

For more information, see **ruth/tools/README.md**.

## Animation
To create animation of the simulation, `FFmpeg` needs to be installed. Using option `--gif` to generate 
gif instead of mp4 does not require `FFmpeg`.

There are two types of animation:
- **Traffic volume animation** - Both color and width of route segments are based on the number of vehicles.
- **Traffic speed animation** - Color is based on the current speed on the segment. Width is based on the number of vehicles.

For more information how to create an animation of an existing simulation, see **ruth/flowmap/README.md**.

# Acknowledgement

* This project has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 957269. 

* Everest project web page: https://everest-h2020.eu

