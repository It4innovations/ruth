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

To perform this task ruth provides tool: `ruth-traffic-flow-to-od-matrix`. For testing purposes you can use files in `ruth/benchmark/hello-world` folder.

```shell
ruth-traffic-flow-to-od-matrix traffic-flow.csv
```

The tool randomly generate GPS points with departure times based on the information withing the traffic flow. As the tool uses random generator to spread points in from/to rectangles and time ranges. The tool also provides information about the border which is an extended convex hull of all generated points. Based on this border is later downloaded a map layer from OSM.

_**NOTE**: currently all records contains the same border. But in the future if distribution of map layers will be considered each record can have its own border_.

### Preprocess traffic simulator input from O/D matrix


## Preparation of probability profile file

- two possibilities: 1) historical data; 2) result of the simulator
