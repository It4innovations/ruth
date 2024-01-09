import os
from collections import defaultdict
from pathlib import Path

import pandas as pd
from ruth.simulator import Simulation
from ruth.vehicle import VehicleAlternatives, VehicleRouteSelection
from flowmap.input import prepare_dataframe
from flowmap.flowmapframe.speeds import segment_speed_thresholds


def sort_df_by_timestamp(df):
    return df.sort_values(by=['timestamp'])


def add_is_first_column(df):
    df = df.sort_values(['vehicle_id', 'timestamp'])
    df['previous_vehicle_id'] = df['vehicle_id'].shift(1).fillna(-1)
    df['is_first'] = df['vehicle_id'] != df['previous_vehicle_id']
    df.drop(columns=['previous_vehicle_id'], inplace=True)
    return df


class SimulationLog:
    def __init__(self, simulation: Simulation):
        self.simulation = simulation
        self.df = prepare_dataframe(simulation.history.to_dataframe(), 1)
        self.df = add_is_first_column(self.df)
        self.df = sort_df_by_timestamp(self.df)
        self.vehicle_alternatives = {}
        self.vehicle_route_selection = {}

        for vehicle in simulation.vehicles:
            self.vehicle_alternatives[vehicle.id] = vehicle.alternatives
            self.vehicle_route_selection[vehicle.id] = vehicle.route_selection

        self.current_timestamp = self.df['timestamp'].iloc[0]

    def get_df_for_next_interval(self, time_interval_minutes: int):
        # current_timestamp = self.df['timestamp'].iloc[0]
        last_timestamp = self.df['timestamp'].iloc[-1]

        def create_df_for_interval():
            # nonlocal current_timestamp
            if self.current_timestamp > last_timestamp:
                return None
            next_timestamp = self.current_timestamp + time_interval_minutes * 60
            current_df = self.df[
                (self.df['timestamp'] >= self.current_timestamp) & (self.df['timestamp'] < next_timestamp)]
            self.current_timestamp = next_timestamp
            return current_df

        return create_df_for_interval

    def get_columns(self):
        columns = [
            'time offset',
            'number of active vehicles in time interval',
            'number of active vehicles in total']
        for alternative in VehicleAlternatives:
            columns.append(f"number of vehicles with {alternative.name.lower()} alternative selection")
        for selection in VehicleRouteSelection:
            columns.append(f"number of vehicles with {selection.name.lower()} route selection")

        columns.extend([
            'number of vehicles that finished journey in time interval',
            'number of vehicles that finished journey in total',
            'meters driven in time interval',
            'meters driven in total',
            'driving time in time interval (minutes)',
            'driving time in total (minutes)',
            'number of segments visited in time interval',
            'number of segments visited in total',
            'average speed in time interval',
            'average speed in total',
        ])

        inf_segment_speed_thresholds = segment_speed_thresholds.copy()
        inf_segment_speed_thresholds = [round(t, 2) for t in inf_segment_speed_thresholds]
        inf_segment_speed_thresholds.append('inf')
        for i in range(len(inf_segment_speed_thresholds) - 1):
            columns.append(
                f"number of segments with average speed in range {inf_segment_speed_thresholds[i]} - {inf_segment_speed_thresholds[i + 1]} km/h in time interval")
        for i in range(len(inf_segment_speed_thresholds) - 1):
            columns.append(
                f"number of segments with average speed in range {inf_segment_speed_thresholds[i]} - {inf_segment_speed_thresholds[i + 1]} km/h in total")
        for i in range(len(inf_segment_speed_thresholds) - 1):
            columns.append(
                f"number of vehicles with average speed in range {inf_segment_speed_thresholds[i]} - {inf_segment_speed_thresholds[i + 1]} km/h in time interval")
        for i in range(len(inf_segment_speed_thresholds) - 1):
            columns.append(
                f"number of vehicles with average speed in range {inf_segment_speed_thresholds[i]} - {inf_segment_speed_thresholds[i + 1]} km/h in total")
        return columns

    def create_log(self, path: str, time_interval_minutes: int):
        df_for_interval = self.get_df_for_next_interval(time_interval_minutes)
        current_df = df_for_interval()
        total_df = pd.DataFrame(columns=current_df.columns)
        with open(path, 'w') as file:
            file.write(';'.join(self.get_columns()) + '\n')
            part_number = 1
            while current_df is not None:
                total_df = pd.concat([total_df, current_df])
                values = []

                # time offset
                values.append(time_interval_minutes * part_number)

                # number of active vehicles in time interval
                values.append(current_df['vehicle_id'].nunique())

                # number of active vehicles in total
                values.append(total_df['vehicle_id'].nunique())

                # number of vehicles with alternative selection
                active_vehicles = current_df['vehicle_id'].unique()
                current_alternatives = defaultdict(int)
                current_route_selection = defaultdict(int)
                for vehicle_id in active_vehicles:
                    current_alternatives[self.vehicle_alternatives[vehicle_id]] += 1
                    current_route_selection[self.vehicle_route_selection[vehicle_id]] += 1

                for alternative in VehicleAlternatives:
                    values.append(current_alternatives[alternative])

                # number of vehicles with route selection
                for selection in VehicleRouteSelection:
                    values.append(current_route_selection[selection])

                # number of vehicles that finished journey in time interval
                values.append(current_df.loc[current_df['active'] == False, 'vehicle_id'].nunique())

                # number of vehicles that finished journey in total
                values.append(total_df.loc[total_df['active'] == False, 'vehicle_id'].nunique())

                # meters driven in time interval
                values.append(round(current_df['meters_driven'].sum()))

                # meters driven in total
                values.append(round(total_df['meters_driven'].sum()))

                # driving time in time interval
                previous_timestamp = self.current_timestamp - time_interval_minutes * 60
                current_df = current_df.sort_values(['vehicle_id', 'timestamp'])
                current_df['timestamp_next'] = current_df['timestamp'].shift(-1)
                current_df.loc[current_df['vehicle_id'] != current_df['vehicle_id'].shift(-1), 'timestamp_next'] = \
                    self.current_timestamp
                current_df.loc[(current_df['vehicle_id'] != current_df['vehicle_id'].shift(-1)) &
                               (current_df['active'] == False),
                               'timestamp_next'] = current_df['timestamp']
                current_df['driving_time'] = current_df['timestamp_next'] - current_df['timestamp']
                current_df['previous_vehicle_id'] = current_df['vehicle_id'].shift(1)
                current_df.loc[(current_df['vehicle_id'] != current_df['previous_vehicle_id']) &
                               (current_df['is_first'] == False), 'driving_time'] += \
                    current_df['timestamp'] - previous_timestamp
                current_df.drop(columns=['timestamp_next', 'previous_vehicle_id'], inplace=True)

                values.append(int(round(current_df['driving_time'].sum() / 60)))
                # set driving time to 1 for vehicles with zero driving time
                # this is done in order to calculate average speed correctly
                current_df.loc[current_df['driving_time'] == 0, 'driving_time'] = 1

                # driving time in total
                total_df_temp = total_df.sort_values(['vehicle_id', 'timestamp'])
                total_df_temp['timestamp_next'] = total_df_temp['timestamp'].shift(-1)
                total_df_temp.loc[
                    total_df_temp['vehicle_id'] != total_df_temp['vehicle_id'].shift(-1), 'timestamp_next'] = \
                    self.current_timestamp
                total_df_temp.loc[(total_df_temp['vehicle_id'] != total_df_temp['vehicle_id'].shift(-1)) &
                                    (total_df_temp['active'] == False),
                                    'timestamp_next'] = total_df_temp['timestamp']
                total_df_temp['driving_time'] = total_df_temp['timestamp_next'] - total_df_temp['timestamp']
                total_df_temp['previous_vehicle_id'] = total_df_temp['vehicle_id'].shift(1)
                total_df_temp.loc[(total_df_temp['vehicle_id'] != total_df_temp['previous_vehicle_id']) &
                                    (total_df_temp['is_first'] == False), 'driving_time'] += \
                    total_df_temp['timestamp'] - previous_timestamp
                total_df_temp.drop(columns=['timestamp_next', 'previous_vehicle_id'], inplace=True)

                values.append(int(round(total_df_temp['driving_time'].sum() / 60)))
                total_df_temp.loc[total_df_temp['driving_time'] == 0, 'driving_time'] = 1

                # number of segments visited in time interval
                values.append(current_df['segment_id'].nunique())

                # number of segments visited in total
                values.append(total_df['segment_id'].nunique())

                # average speed in time interval
                current_df['weighted_speed'] = current_df['speed_mps'] * current_df['driving_time']
                current_df = current_df.loc[current_df['driving_time'] > 0]
                current_avg_speed = current_df['weighted_speed'].sum() / current_df['driving_time'].sum()
                values.append(str(round(current_avg_speed, 2)).replace('.', ','))

                # average speed in total
                total_df_temp['weighted_speed'] = total_df_temp['speed_mps'] * total_df_temp['driving_time']
                total_df_temp = total_df_temp.loc[total_df_temp['driving_time'] > 0]
                total_avg_speed = total_df_temp['weighted_speed'].sum() / total_df_temp['driving_time'].sum()
                values.append(str(round(total_avg_speed, 2)).replace('.', ','))

                # number of segments with average speed in range in time interval
                df_segment_speeds = current_df.groupby('segment_id').agg({'weighted_speed': 'sum',
                                                                          'driving_time': 'sum'})
                # df_segment_speeds = df_segment_speeds[df_segment_speeds['driving_time'] > 0]
                df_segment_speeds['average_speed'] = \
                    df_segment_speeds['weighted_speed'] / df_segment_speeds['driving_time']

                inf_segment_speed_thresholds = segment_speed_thresholds.copy()
                inf_segment_speed_thresholds.append(10000)
                for i in range(len(inf_segment_speed_thresholds) - 1):
                    values.append(
                        len(df_segment_speeds.loc[
                                (df_segment_speeds['average_speed'] >= inf_segment_speed_thresholds[i]) &
                                (df_segment_speeds['average_speed'] < inf_segment_speed_thresholds[i + 1])]))

                # number of segments with average speed in range in total
                df_segment_speeds_total = total_df_temp.groupby('segment_id').agg({'weighted_speed': 'sum',
                                                                                    'driving_time': 'sum'})
                # df_segment_speeds_total = df_segment_speeds_total[df_segment_speeds_total['driving_time'] > 0]

                df_segment_speeds_total['average_speed'] = \
                    df_segment_speeds_total['weighted_speed'] / df_segment_speeds_total['driving_time']

                for i in range(len(inf_segment_speed_thresholds) - 1):
                    values.append(
                        len(df_segment_speeds_total.loc[
                                (df_segment_speeds_total['average_speed'] >= inf_segment_speed_thresholds[i]) &
                                (df_segment_speeds_total['average_speed'] < inf_segment_speed_thresholds[i + 1])]))

                # number of vehicles with average speed in range in time interval
                df_vehicle_speeds = current_df.groupby('vehicle_id').agg({'weighted_speed': 'sum',
                                                                          'driving_time': 'sum'})
                # df_vehicle_speeds = df_vehicle_speeds[df_vehicle_speeds['driving_time'] > 0]

                df_vehicle_speeds['average_speed'] = \
                    df_vehicle_speeds['weighted_speed'] / df_vehicle_speeds['driving_time']

                for i in range(len(inf_segment_speed_thresholds) - 1):
                    values.append(
                        len(df_vehicle_speeds.loc[
                                (df_vehicle_speeds['average_speed'] >= inf_segment_speed_thresholds[i]) &
                                (df_vehicle_speeds['average_speed'] < inf_segment_speed_thresholds[i + 1])]))

                # number of vehicles with average speed in range in total
                df_vehicle_speeds_total = total_df_temp.groupby('vehicle_id').agg({'weighted_speed': 'sum',
                                                                                    'driving_time': 'sum'})
                # df_vehicle_speeds_total = df_vehicle_speeds_total[df_vehicle_speeds_total['driving_time'] > 0]
                df_vehicle_speeds_total['average_speed'] = \
                    df_vehicle_speeds_total['weighted_speed'] / df_vehicle_speeds_total['driving_time']

                for i in range(len(inf_segment_speed_thresholds) - 1):
                    values.append(
                        len(df_vehicle_speeds_total.loc[
                                (df_vehicle_speeds_total['average_speed'] >= inf_segment_speed_thresholds[i]) &
                                (df_vehicle_speeds_total['average_speed'] < inf_segment_speed_thresholds[i + 1])]))

                # add values to the csv
                file.write(";".join([str(value) for value in values]) + "\n")

                current_df = df_for_interval()
                part_number += 1


def create_simulation_log(simulation: Simulation, output_path: str, time_interval_minutes: int):
    simulation_log = SimulationLog(simulation)
    simulation_log.create_log(output_path, time_interval_minutes)


def create_simulations_comparison(simulation_paths: list[str], output_dir: str, time_interval_minutes: int):
    for path in simulation_paths:
        simulation = Simulation.load(path)
        output_csv_path = os.path.join(output_dir, Path(path).stem + ".csv")
        create_simulation_log(simulation, output_csv_path, time_interval_minutes)
