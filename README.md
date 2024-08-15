# ruth

<p align="center">
    <img width="60%" src="docs/images/ruth.svg?sanitize=true">
</p>

A python library for routing on OSM map based on [osmnx](https://github.com/gboeing/osmnx).

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
    ruth-simulator --departure-time="2021-06-16 07:00:00" --k-alternatives=4 --out=simulation_record.pickle --seed=7 run INPUT-od-matrix-10-vehicles-town-resolution.parquet  --alt-dijkstra-fastest=0.3 --alt-plateau-fastest=0.0 --selection-first=0.3 --selection-random=0.0 --selection-ptdr=0.0
    ```
* input data for 10K-300K vehicle simulations can be found in this [dataset](https://doi.org/10.5281/zenodo.13285293).
### Distributed run
More information about distributed run can be found in **ruth/zeromq/README.md**.

### Options
For the Alternatives and Route selection, percentages of vehicles can be set for each type. The sum of percentages for Alternatives has to be equal to the sum for Route Selection. If the percentages don't add up to 1, no alternatives are calculated for the remaining vehicles and they stick to their original route that is in the input parquet file.
#### Alternatives
- shortest-paths networkx implementation of dijkstra algorithm using route length as weight
- fastest-paths: networkx implementation of dijkstra algorithm using current-travel-time as weight
- distributed: cpp implementation of Plateau algorithm
  - provided by [evkit](https://code.it4i.cz/everest/evkit)
  - ports either taken from the environment or set to 5555 and 5556.
  - **plateau-default-route**: flag to recalculate the default route with Plateau algorithm
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
- then, use [this code](https://code.it4i.cz/mic0427/ptdr/) (a currently private repository - access may be granted upon request) to calculate probability profiles (json representation of msqpack file below):
  
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
#### Stuck detection
Parameter `stuck-detection` defines the number of round-frequency-s long rounds with no vehicles movement
that have to pass before the simulation is terminated.  
The detection is based on all vehicles being in the same position for the whole time no temporary max speeds are to be updated.
If the parameter is **set to 0**, the detection is **disabled**.

### Animation
To create animation of the simulation after the run, `FFmpeg` needs to be installed. Using option `--gif` to generate 
gif instead of mp4 does not require `FFmpeg`.

There are two types of animation:
- **Traffic volume animation** - Both color and width of route segments are based on the number of vehicles.
- **Traffic speed animation** - Color is based on the current speed on the segment. Width is based on the number of vehicles.

To create the speed animation using the configuration file, run:
``` sh
ruth-simulator-conf --config-file="config.json" run speed-animation
```
for the volume animation, run:
``` sh
ruth-simulator-conf --config-file="config.json" run volume-animation
```
Animation can be generated only after `run` command. 
To create an animation of an existing simulation, use [flowmap package](https://code.it4i.cz/mic0427/traffic-flow-map).

Animation options can be set in configuration file, for example:
``` json
{
  ...
  "animation":
  {
    "length": 60,
    "gif": true,
    "fps": 20,
    "title": "Traffic speed animation"
  }
}
```

#### Animation options
- **fps** - frames per second (default 25)
- **save_path** - path to the folder for the output video (default current directory)
- **frame_start** - number of frames to skip (default 0)
- **frames_len** - number of frames to plot (default all)
- **width_modif** - adjust width of the segments (default 10, select values in range 2 - 200)
- **title** - video title (default empty)
- **description** - description to be added to the video, overrides `description_path` option (default empty)
- **description_path** - path to the file with description to be added to the video, only used if `description` option is not set
- **length** - video length in seconds (default 30)
- **divide** - into how many parts will each segment be split (default 2)
- **max_width_count** - number of vehicles that corresponds to the maximum width of the segment, if not specified, it will be set dynamically according to the data
  - important to set when creating multiple animations to compare
- **plot_cars** - visualize cars on the map
- **zoom** - choose zoom manually
- **gif** - generate as a gif instead of mp4, does not require `FFmpeg`



# Acknowledgement

* This project has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 957269. 

* Everest project web page: https://everest-h2020.eu

