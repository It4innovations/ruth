# Available tools


Apart from the simulator itself there is several other tools provided.

## Tools provided by Ruth

- `ruth-simulator` - deterministic traffic simulator
- `ruth-traffic-flow-to-od-matrix` - generate origin-destination (O/D) matrix from traffic flow description (details in `help`; the traffic flow file is prepared by Radek Furmanek.)
- `ruth-od-matrix-to-simulator-input` - generate from O/D matrix the input for the simulator. It may take time, as it pins the GPS positions to map segments. Embarrassingly parallel, the more cores (threads) provided the quicker processing. Ideally let it compute on Karolina node.
- `ruth-aggregate-data` - aggregate the history records. The aggregated version can be used as an input for generating probability profiles with `process-time-range-histograms-from-csv` (FCD History Analytics bellow)
- `ruth-show-gv` - only for debug purposes it shows a map with dots representing the FCD at the particular time slot. Not suitable for huge history records; only for debug. It might be used as an inspiration how to look at the data.
- `ruth-split-segment-id` - **obsolete!** It was used to prepare data for flow map visualisation (Pavlina Smolkova & Sofia Michailidu).
- `ruth-data-preprocessing` - **obsolete!** the original input data preparation; not valid anymore.


## Tools provided by FCDHistoryAnalytics component (Rust)

- https://code.it4i.cz/everest-internal/fcdhistoryanalytics (public access token: [[https://project_1113_bot:v4GzuGgiqjycBqCAy4ms@code.it4i.cz/everest/fcdhistoryanalytics.git]])

- The idea behind, was to create a toolset with appropriate data structures that is able to process historical FCD data provided by ŘSD (ředitelství silnic a dálnic).
- Project divided into 4 packages:
	- `data` - responsible for generating **Time Range Histograms** (TRH) from FCD data. It can connect to IT4I's database or it can be provided with CSV file; CSV file is suitable for generating the TRHs from simulators output.
	- `datatimebox` - data structures related to time and space. Providing functionality over such times, like iterators, hashing, etc.
		- `timerange` - time range defined by start, end, and step 
		- `timeindex` - generic definition of time index and its generator. Simulator then uses so called **Group Time Index**. It divides a time range into groups and each group is then represented by its index.
		- `storages/timerangehistogram` - a storage which stores a time range histogram for particular time index and *space*. The *space* refers to location on map, e.g. routing segment.
	- `ptdr` - functionalities and data structures related to Probabilistic Time Dependent Routing (PTDR); More precisely there is covered only part of Monte Carlo simulation (**a procedure computing probable delay on a route at departure time**). The other part is related to so called **probability profiles**. A probability profile is a simplified view on distribution of *level of service* (LoS) on route segment in time. The distribution is given by TRH. The TRH is then sampled (in our case to quartiles + 99 percentile) and these samples form one record of probability profile.
	- `scripts` - python script for generating so called **Delay Graph** (the graph shown in gitlab readme page)
- There are other EVEREST related packages:
	- `ptdr-cpp` - distilled version of Monte Carlo written in C++ and suitable for HW synthesis. The probability profiles are attached to each segment because of FPGA. It is clear that is inefficient but it's just for demo purpose of working FPGA version (up to this point 14.2.2023 still not available). For the future version is expected that probability profiles will be stored in memory "near to FPGA" with random access.
	- other scripts in `scripts` folder
- The last folder `testing-data` contains a basic of data for testing

### Build
- In root directory call `cargo build --release`. Be careful debug version is much slower.
- The build tools are located in `./target/release` folder.

### Workflow
1. Generate a set of time range histograms for each segment at time index.
2. From the previous step generate probability profiles. Later used in simulator or for Monte Carlo simulation (computation of probable delay on route at departure time).

### Tools

#### Data package
- `process_time_range_historgrams` - gather the info from database; it is necessary to specify a connection to IT4I's historical data (Contact: Radek Furmanek). Nevertheless, currently useless as the historical data uses different indexing of segments (not connectable to OSM data used by simulator). **It will be useful once we have a relation between TMC segments and OSM map network; or we get maps which works with TMC segments - maps provided by CEDA.**
- `process_time_range_histograms_from_csv`
	- example: `./target/release/process_time_range_histograms_from_csv ~/projects/everest/ruth-cluster-data/aggregated-fcd-sim.csv "2021-06-20 23:59" --out simulator-data`
- Helper tools:
	- `joiner` - joins two sets of time range histograms into one file
	- `cmp_histograms` - a histogram comparator for debug purposes 
	- `show_histogram` - a histogram plotter

#### Ptdr package
- `time_range_histograms_to_probab_profiles` sample the distributions of LoS on route segments at time index and produce probability profile file used by the simulator.
	- example: `./target/release/time_range_histograms_to_prob_profiles histograms-for-2021-06-20T23:59:00+02:00--2021-06-27T23:59:00+02:00.data`
- `probdelay` a binary which can compute a probable delay on a route at departure time. It Can also produce these info for range of times, e.g., the entire day with 5min step. This can be then processed with `scripts/delay-graph.py`. 

