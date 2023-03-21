# Data preparation

In order to run the simulator we need to prepare two crucial input files:

1. Origin-destination matrix
2. probability profiles

## Preparation of O/D matrix as input of the traffic simulator

To prepare Origin-destination matrix as an input of traffic simulator is two steps process. Firstly, the O/D matrix from traffic flow description is generated. Then the O/D matrix is "initialized", i.e., the GPS points are attached to route segments in map, and the initial shortest path for each record is computed.

### Generating O/D matrix
The first input of the traffic simulator is an Origin-destination matrix. To generate it we need a description of _traffic flow_. This file is provided by [Radek Furmánek](<mailto:radek.furmanek@vsb.cz>) or for testing purposes it might be generated as well.

The traffic flow is stored in form of CSV with the following columns:
  - `start_time`: a window start time
  - `end_time`: a window end time
  - `count_devices`: a number of devices going from the origin rectangle to destination one.
  - `geom_rectangle_from`: geometry of origin rectangle
  - `geom_rectangle_to`: geometry of destination rectangle

The flow is defined by transitions between areas of predefined size (`rectangle_from`, `rectangle_to`). From the flow we just know that there is a specific amount of "devices"/vehicles (`count_devices`) which go from one rectangle to another at specific time range defined by `start_time` and `end_time`. _Importantly, the `end_time` is not the time  when the vehicle reaches the destination rectangle, but the end of the time window in which the vehicle departs!_

#### Tool
To perform this task `ruth` provides tool: `ruth-traffic-flow-to-od-matrix`. For testing purposes you can use files in `ruth/benchmark/hello-world` folder.

```shell
ruth-traffic-flow-to-od-matrix traffic-flow.csv --out od-matrix.csv
```

#### Output description

The tool randomly generate GPS points with departure times based on the information withing the traffic flow. As the tool uses random generator to spread points in from/to rectangles and time ranges. The tool also provides information about the border which is an extended convex hull of all generated points. Based on this border is later downloaded a map layer from OSM.

_**NOTE**: currently all records contains the same border. But in the future if distribution of map layers will be considered each record can have its own border_.

### Preprocess traffic simulator input from O/D matrix

The second step is preprocessing of the O/D matrix to the simulator's input. The input is serialized version of `pandas.DataFrame` object. This actually represents a state of all the vehicles. 

The preprocessing stage consists of three steps:

1. initialization of `Vehicle`s state,
2. attach GPS positions to map segments, i.e., transform latitude and longitude of origin and destination GPS points to origin and destination _nodes_,
3. compute initial shortest path between `origin` and `destinaion` nodes.

The second and third steps are performance demanding and the preprocessing tool is prepared for it. It offers thread based parallelization. The task is embarrassingly parallel, hence use as much resources as you can. The argument that enables this is `--nproc`; the default is 1.

#### Tool

To perform the preprocessing step `ruth` provides tool: `ruth-od-matrix-to-simulator-input`. As the input use a file generated in previous step or `benchmark/hello-world/od-matrix.csv`. Please note, that the files may differ as there is used the random generator in the previous step.

```shell
ruth-od-matrix-to-simulator-input od-matrix.csv --out vehicles-state.parquet --nproc=8
```

The `vehicles-state.parquet` is the first input of `ruth-simulator`.

## Preparation of probability profile file

There are two possibilities to prepare the probability profile files. The first is to use historical data collected and managed by IT4Innovation. For the information about connection please contact [Radek Furmánek](<mailto:radek.furmanek@vsb.cz>).

The tool providing an intermediate step is called `process_time_range_histograms`, and it is available in [FCDHistoryAnalytics](https://code.it4i.cz/everest/fcdhistoryanalytics) project. This is a library written in Rust providing functionality over historical data and performance demanding parts of _Probabilistic Time Dependent Routing_ (PTDR).

**Disclaimer:** _At this point, this approach is not usable in the context of traffic simulator. The simulator uses a Open Street Maps and the date provided by ŘSD use different segmentation and route segment indexing._

To have a simulation result to process we need firstly to run the simulator.
```shell
ruth-simulator --departure-time="2021-06-16 07:00:00" --k-alternatives=4 --nproc=8 --out=simulation-result.pickle --seed=7 rank-by-prob-delay vehicles-state.parquet 70 500
```

### Generating probability profile based on simulation result.

The second option how to get a probability profile is to use the result of simulation.

This a three-step process:

1. aggregating FCD based on time
```shell
ruth-aggregate-fcd aggregate-globalview simulation-result.pickle --round-freq-s 15 --out aggregated_fcd.csv
```

2. compute time range histograms
```shell
process_time_range_histograms_from_csv aggregated_fcd.csv "2021-06-20 23:59" --out time_range_histograms
```
The date time is the end of time window. The probability profile is typically computed for one week, i.e., end of the week in which the simulation was performed. For further setting use `--help`.

3. generate probability profiles
```shell
time_range_histograms_to_prob_profiles time_range_histograms.data
```

The second and third tool are from [FCDHistoryAnalytics](https://code.it4i.cz/everest/fcdhistoryanalytics).



