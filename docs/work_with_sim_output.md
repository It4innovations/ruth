# Simulation output

The output of the simulation is a serialized object of `ruth.simulator.Simulation` type. The resulting object represents a captured state of the simulation from which it can continue; if not finished yet. Apart from the evolution of traffic during the simulation, `Simulation` contains other useful information and post processing functions, e.g., performance inside, post-processing data in form of `pandas.DataFrame`, etc.

## Read the simulation

To look at the simulation results one can write a Python script or use interactive shell.

```jupyterpython
from ruth.simulator import Simulation

sim = Simulation.load("/path/to/sim-result.pickle")
```

### History
The result of simulation is recorded in `history`. The history can be processed as a `pandas.DataFrame.

```jupyterpython
history_df = sim.history.to_dataframe()
```

|    | timestamp                  | segment_id              |   vehicle_id |   start_offset_m |   speed_mps |   segment_length | status      |   node_from |    node_to |
|---:|:---------------------------|:------------------------|-------------:|-----------------:|------------:|-----------------:|:------------|------------:|-----------:|
|  0 | 2021-06-16 07:00:27.331000 | OSM31765033T2671968558  |            3 |          19.425  |     8.33333 |          218.052 | not started |    31765033 | 2671968558 |
|  1 | 2021-06-16 07:00:27.820000 | OSM265825223T73388626   |            1 |          15.1667 |     8.33333 |           67.275 | not started |   265825223 |   73388626 |
|  2 | 2021-06-16 07:00:30        | OSM10288560273T31764596 |            4 |          69.4444 |    13.8889  |          118.922 | not started | 10288560273 |   31764596 |

A single record in history consists of:

* `timestamp` - timestamp of an event,
* `segment_id` - a unique identifier of route segment,
* `vehicle_id` - a unique identifier of vehicle,
* `start_offset_m` - an offset in meters from the beginning of segment (`node_from`),
* `speed_mps` - a speed assigned to the vehicle at this point of time in m/s,
* `segment_length` - a length of the segment,
* `node_from` - a starting node of the segment,
* `node_to` - an ending node of the segment,
* `status` - an arbitrary text used for debugging purposes (_"not started"_ is ok status anything else is suspicious).

### Global view

```jupyterpython
gv_df = sim.global_view.to_dataframe()
```

The globalview is a subset of history records. The history records are only collected via the simulation and it is an actual result used for further analysis. On the other hand, _global view_ is an active subset of history, i.e., list of records that are used for computation level of service in a _near distance_ ("a distance driver see directly from window"; it's a hyperparameter of the simulation). It's not valid for any kind of analysis. The global view can be gained from history by calling `drop_old` method. All records older then provided threshold are droped.

## Performance insight

During the simulation duration of particular functions is measured. This is stored in `step_info`, and similarly to `history` it can be also processed as dataframe.

```jupyterpython
perf_df = sim.step_info_to_dataframe()
```

|    |   step |   n_active |   duration |   allowed_vehicles |   alternatives |    collect |   vehicle_plans |   select_plans |   transform_plans |   advance_vehicle |    update |   compute_offset |   drop_old_records |   end_step |
|---:|-------:|-----------:|-----------:|-------------------:|---------------:|-----------:|----------------:|---------------:|------------------:|------------------:|----------:|-----------------:|-------------------:|-----------:|
|  0 |      0 |          1 |    2272.18 |          0.146151  |        2267.79 | 0.00524521 |      0.0038147  |        3.75414 |         0.211954  |         0.157356  | 0.0331402 |        0.0119209 |         0.00405312 | 0.00190735 |
|  1 |      1 |          3 |    3144.36 |          0.0708103 |        3137.53 | 0.0100136  |      0.00691414 |        6.30188 |         0.136137  |         0.220299  | 0.0469685 |        0.0140667 |         0.00500679 | 0.0038147  |
|  2 |      2 |          1 |    2227.49 |          0.0720024 |        2225.63 | 0.00691414 |      0.00286102 |        1.60289 |         0.0360012 |         0.0760555 | 0.0309944 |        0.0119209 |         0.00476837 | 0.00119209 |

A single record in step info consists of:

* `step` - a step number,
* `n_active` - a number of active vehicles; those which moved in the step,
* `duration` - an overall duration of the step in milliseconds,
* `allowed_vehicles` - a duration of filtering which vehicles can move,
* `alternatives` - a duration of computation alternative routes,
* `collect` - coleecting vehicles without alternatives and finishing them,
* `vehicle_plans` - a duration to assemble all possible plans for vehicle,
* `select_plans` - a duration of process of picking the plan which will be used; here is the Monte Carlo simulation used for estimation of probable delay on a route,
* `transform_plans` - a plan consist of vehicle and `Route` used for simulation; it transform the `Route` to OSM route,
* `advance_vehicle` - a duration of moving of a vehicle,
* `update` - update the simulation state,
* `compute_offset` - compute the next minimal offset for selecting next bunch of active vehicles,
* `drop_old_records` - a duration of dropping records within **global view**,
* `end_step` - a duration of user specific function performed at the end of each step.


## Look at caches
```jupyternotebook

# find out used caches
sim.caches
# defaultdict(<function ruth.simulator.simulation.get_lru_cache()>,
#           {'alternatives': <pylru.lrucache at 0x16dfcefd0>})

caches_df = sim.cache_info_to_dataframe('alternatives')
```

|    | timestamp                  |   n_hits |   total |   hit_rate |
|---:|:---------------------------|---------:|--------:|-----------:|
|  0 | 2023-03-23 10:11:39.678220 |        0 |       1 |          0 |
|  1 | 2023-03-23 10:11:41.908312 |        0 |       3 |          0 |
|  2 | 2023-03-23 10:11:45.052772 |        0 |       1 |          0 |

A single record in cache info consists of:

* `timestamp` - a timestamp of event,
* `n_hits`  - how many times the cache was hit,
* `total` - a number of vehicles for which the alternatives are computed,
* `hit_rate` - `n_hits` / `total`.



# Simulation postprocessing

The records stored in history can be considered as so-called RAW FCD (points in time on map). In order to get aggregated version the simulation can be postprocessed via `ruth-aggregate-fcd` tools. There are two versions `aggregate-globalview` and `aggregate-globalview-set`. The second one can be used to aggregate the results of more than one simulation. By aggregating one simulation results we get a CSV of the following format:

```jupyternotebook
ruth-aggregate-fcd aggregate-globalview  --round-freq-s 300 --out aggregated_fcd.csv
```

|    | segment_osm_id         | fcd_time_calc    |        los |   segment_length |
|---:|:-----------------------|:-----------------|-----------:|-----------------:|
|  0 | OSM236504238T694219    | 2021-06-16 07:20 | 0.928181   |          373.471 |
|  1 | OSM29745937T29745938   | 2021-06-16 07:20 | 0          |           43.018 |
|  2 | OSM2506908136T26424525 | 2021-06-16 07:15 | 0.00741471 |           48.714 |

A single record consists of:
* `segment_osm_id` - an id of OSM segment,
* `fcd_time_calc` - a calculated time based `-round-freq-s`,
* `los` - level of service on segment
* `segment_length` - length of the segment

These data can be used for further analysis. For example, we can define something like traffic jam rate:

```jupyternotebook

import pandas as pd

df = pd.read_csv("aggregated_fcd.csv", sep=';')

tj_rate = len(df[df["los"] < 0.2]) / len(df)
```

If the level of service is bellow 20% we can consider it as a traffic jam. The definition of threshold may differ.
 
