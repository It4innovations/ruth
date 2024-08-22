# Speed Profiles
Script for aggregation of records from [Ruth](https://github.com/It4innovations/ruth) simulation.

Only records for timestamps and segments which are present in the simulation are considered (segments with no vehicles are not present in the output).

## Install
1. Install [Ruth](https://github.com/It4innovations/ruth)
2. Install packages from `requirements.txt`:

## Run
```
python3 speed_profiles.py --help
```

### Arguments
* path to the simulation record file,
* `--round-freq-s` - frequency of the aggregation in seconds,
* `--out` - path to the output file.

## Example
```
python3 speed_profiles.py simulation_record.pickle --round-freq-s 600 --out speed_profiles.csv
```
This command will create csv file with aggregation for 10 minutes intervals.

## Output
The output is a CSV file with the following columns:
* `date`,
* `road_id` - an id of OSM segment in format `OSM<from_node_id>T<to_node_id>`,
* `time_in_minutes_from` - record start time in minutes from midnight of the day in the `date` column,
* `time_in_minutes_to` - record end time in minutes from midnight of the day in the `date` column,
* `speed_kph` - average speed in km/h