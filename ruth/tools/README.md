# Tools
This directory contains scripts for preprocessing and postprocessing of data related to the Ruth simulation.

## Traffic Flow to OD Matrix
This tool creates Origin Destination matrix needed for generating simulator input parquet file. 
It generates a CSV file with one row per vehicle.
### Input
Input is a traffic flow description CSV. Each row represents one group of vehicles.  
CSV contains columns:
- `start_time`- departure window start time
- `end_time`- departure window end time
- `count_devices` - number of devices/vehicles going from the origin to the destination
- `geom_rectangle_from` - geometry of the origin rectangle
- `geom_rectangle_to` - geometry of the destination rectangle

```
count_devices,start_time,end_time,geom_rectangle_from,geom_rectangle_to
15,2021-06-16 08:00:00+02,2021-06-16 08:15:00+02,"POLYGON((14.420298576355 50.0022201538086,14.420298576355 50.006721496582,14.4271478652954 50.006721496582,14.4271478652954 50.0022201538086,14.420298576355 50.0022201538086))","POLYGON((14.4408464431763 49.9977149963379,14.4408464431763 50.0022201538086,14.4476947784424 50.0022201538086,14.4476947784424 49.9977149963379,14.4408464431763 49.9977149963379))"
```
Rectangles are specified by their corner points. Format:   
POLYGON((<lon_min> <lat_min>,<lon_min> <lat_max>,<lon_max> <lat_max>,<lon_max> < lat_min>,<lon_min> <lat_min>))

### Example
```
ruth-traffic-flow-to-od-matrix traffic_flow.csv --out od_matrix.csv
```

### Output
Output is a CSV file with one row per vehicle. For each of them CSV specifies:
- `id` - vehicle ID
- `lat_from`,`lon_from`,`lat_to`,`lon_to` - origin and destination coords randomly selected in the specified rectangles
- `start_offset_s` - departure time offset in seconds randomly generated according to the length of the selected time window

Note: Departure time of a vehicle in the OD matrix is specified only by the offset in seconds and there is no information about the particular timestamp. Therefore departure time in the traffic flow doesn't have to match departure time later selected as a simulator setting.

## Combine OD matrices
This command enables combining multiple origin destination matrices csv files into one csv file.

### Input
The OD matrices are combined based on the rules. The rules are a csv file with the following format:
- `timestamp` - time when the OD matrix should be used - offset since midnight  
- `swap` - boolean value which indicates if the origin and destination should be swapped
-  `path` - path to the OD matrix csv file

```csv
timestamp;swap;path
7:00:00;0;5K.csv
12:35:00;1;10K.csv
```

### Example
```
ruth-combine-od-matrix rules.csv --separator ';' --out combined_od.csv
```

## OD Matrix to Simulation Input
This command creates a simulator input parquet file from the OD matrix file.
### Example
```
ruth-od-matrix-to-simulator-input od_matrix.csv --out simulation_input.parquet
```

### Arguments:
- `--download-date` - the date for which the map is downloaded
- --increase-lat, --increase-lon - a fraction of the area increase in the map
- `--lat-min`, `--lat-max`, `--lon-min`, `--lon-max` - can be set to increase the map size to specific lat/lon values
- `--csv-separator`
- `--frequency` - a period in which a vehicle asks for rerouting in seconds (default is 20 s)
- `--fcd-sampling-period` - a period in which FCD is stored, it sub samples the frequency (default is 5 s)
- `--nproc` - number of used processes
- `--data-dir` - directory to save map graphml
- `--out` - output parquet name
- `--show-only` - flag to show map with cars without computing output parquet file

Tool calculates the map borders. It takes the rectangle where the vehicles have their origin and destination points. In case of lat/lon min/max options enlarging the map, rectangle border is moved according to this value. This setting cannot make the map smaller, all the origin and destination points are always in the map area. This rectangle is then enlarged according to the increase lat/lon options.   

For each vehicle, the closest origin and destination OSM node is chosen for both the origin and destination location (defined in lat lon coordinates in the input csv). 
Then OSM route is calculated and saved as list of OSM node IDs. Currently Dijkstra fastest paths algorithm is used (implemented by networkx library). This route can then be recalculated at the beginning of the simulation by Plateau.  

### Output
Output is a pandas dataframe saved in parquet file. This dataframe consists of:
- Columns with data taken from osm route calculation (different for each vehicle): `id`, `origin_node`, `dest_node`, `time_offset`, `osm_route`, `start_index`
- Columns set to zero (same for each vehicle): `start_index`, `start_distance_offset`,
- Columns taken from parameters (same for each vehicle):  `frequency`, `fcd_sampling_period`,
- Columns defining the map (same for each vehicle): `download_date`, `bbox_lat_max`, `bbox_lon_min`, `bbox_lat_min`, `bbox_lon_max`
- Columns describing the state of the vehicle: `active`,`status`


## Aggregated FCD
Command to generate file containing speeds on segments in time slots. The computation is based on existing simulation records. Speeds are calculated as an average of recorded speeds or max speed is used in case of no vehicles present.
### Example
```
ruth-aggregate-fcd aggregate-globalview simulation_record.pickle --round-freq-s 300 --out speed_profiles.csv
```
### Arguments
`--round-freq-s` defines how to round timestamps in seconds (time slot length)
`--out` output file

### Bulk aggregation
Make aggregation file for each simulation in a specified directory
```
ruth-aggregate-fcd aggregate-globalview-set <DIR_PATH> --round-freq-s 600 --out-dir <OUT_DIR>
```
### Output
The output is a CSV file with the following columns:  
- `segment_osm_id` - segment OSM id  
- `fcd_time_calc` - rounded timestamp  
- `segment_length` - length of the segment in meters  
- `max_speed `- max speed of the segment  
- `current_speed` – averaged vehicle’s speed on segment at time   

## Speed Profiles
Command for aggregation of records from Ruth simulation.

Only records for timestamps and segments which are present in the simulation are considered (segments with no vehicles are not present in the output).

### Arguments
* path to the simulation record file,
* `--round-freq-s` - frequency of the aggregation in seconds,
* `--out` - path to the output file.

### Example
```
ruth-speed-profiles simulation_record.pickle --round-freq-s 600 --out speed_profiles.csv
```
This command will create csv file with aggregation for 10 minutes intervals.

### Output
The output is a CSV file with the following columns:
* `date`,
* `road_id` - an id of OSM segment in format `OSM<from_node_id>T<to_node_id>`,
* `time_in_minutes_from` - record start time in minutes from midnight of the day in the `date` column,
* `time_in_minutes_to` - record end time in minutes from midnight of the day in the `date` column,
* `speed_kph` - average speed in km/h