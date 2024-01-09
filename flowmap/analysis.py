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
    def __init__(self, simulation: Simulation, time_interval_minutes: int):
        self.simulation = simulation
        self.time_interval_minutes = time_interval_minutes
        self.df = prepare_dataframe(simulation.history.to_dataframe(), 1)
        self.df = add_is_first_column(self.df)
        self.df = sort_df_by_timestamp(self.df)
        self.vehicle_alternatives = {}
        self.vehicle_route_selection = {}

        for vehicle in simulation.vehicles:
            self.vehicle_alternatives[vehicle.id] = vehicle.alternatives
            self.vehicle_route_selection[vehicle.id] = vehicle.route_selection

        self.current_timestamp = self.df['timestamp'].iloc[0]

    def get_df_for_next_interval(self):
        last_timestamp = self.df['timestamp'].iloc[-1]

        def create_df_for_interval():
            if self.current_timestamp > last_timestamp:
                return None
            next_timestamp = self.current_timestamp + self.time_interval_minutes * 60
            current_df = self.df[
                (self.df['timestamp'] >= self.current_timestamp) & (self.df['timestamp'] < next_timestamp)]
            self.current_timestamp = next_timestamp
            return current_df

        return create_df_for_interval

    def get_columns(self):
        columns = [
            'time offset (minutes)',
            'number of active vehicles in time interval',
            'number of active vehicles since start']
        for alternative in VehicleAlternatives.__members__.values():
            columns.append(f"number of active vehicles with {alternative.name.lower()} alternative selection")
        for selection in VehicleRouteSelection.__members__.values():
            columns.append(f"number of active vehicles with {selection.name.lower()} route selection")

        columns.extend([
            'number of vehicles that finished journey in time interval',
            'number of vehicles that finished journey since start',
            'meters driven in time interval',
            'meters driven since start',
            'driving time in time interval (minutes)',
            'driving time since start (minutes)',
            'number of segments visited in time interval',
            'number of segments visited since start',
            'average speed in time interval',
            'average speed since start',
        ])

        inf_segment_speed_thresholds = segment_speed_thresholds.copy()
        inf_segment_speed_thresholds = [round(t, 2) for t in inf_segment_speed_thresholds]
        inf_segment_speed_thresholds.append('inf')
        for i in range(len(inf_segment_speed_thresholds) - 1):
            columns.append(
                f"number of segments with average speed in range {inf_segment_speed_thresholds[i]} - {inf_segment_speed_thresholds[i + 1]} km/h in time interval")
        for i in range(len(inf_segment_speed_thresholds) - 1):
            columns.append(
                f"number of segments with average speed in range {inf_segment_speed_thresholds[i]} - {inf_segment_speed_thresholds[i + 1]} km/h since start")
        for i in range(len(inf_segment_speed_thresholds) - 1):
            columns.append(
                f"number of vehicles with average speed in range {inf_segment_speed_thresholds[i]} - {inf_segment_speed_thresholds[i + 1]} km/h in time interval")
        for i in range(len(inf_segment_speed_thresholds) - 1):
            columns.append(
                f"number of vehicles with average speed in range {inf_segment_speed_thresholds[i]} - {inf_segment_speed_thresholds[i + 1]} km/h since start")
        return columns

    def get_vehicle_types_counts(self, df):
        active_vehicles = df['vehicle_id'].unique()
        alternatives = defaultdict(int)
        route_selection = defaultdict(int)
        for vehicle_id in active_vehicles:
            alternatives[self.vehicle_alternatives[vehicle_id]] += 1
            route_selection[self.vehicle_route_selection[vehicle_id]] += 1
        return alternatives, route_selection

    def calculate_driving_time(self, df):
        previous_timestamp = self.current_timestamp - self.time_interval_minutes * 60
        df = df.sort_values(['vehicle_id', 'timestamp'])
        df['timestamp_next'] = df['timestamp'].shift(-1)
        df.loc[df['vehicle_id'] != df['vehicle_id'].shift(-1), 'timestamp_next'] = \
            self.current_timestamp
        df.loc[(df['vehicle_id'] != df['vehicle_id'].shift(-1)) & (df['active'] == False), 'timestamp_next'] = \
            df['timestamp']
        df['driving_time'] = df['timestamp_next'] - df['timestamp']
        df['previous_vehicle_id'] = df['vehicle_id'].shift(1)
        df.loc[(df['vehicle_id'] != df['previous_vehicle_id']) & (df['is_first'] == False), 'driving_time'] += \
            df['timestamp'] - previous_timestamp
        df.drop(columns=['timestamp_next', 'previous_vehicle_id'], inplace=True)
        return df

    def add_segment_speed_values(self, df, values):
        df_segment_speeds = df.groupby('segment_id').agg({'weighted_speed': 'sum', 'driving_time': 'sum'})
        df_segment_speeds['average_speed'] = df_segment_speeds['weighted_speed'] / df_segment_speeds['driving_time']
        inf_segment_speed_thresholds = segment_speed_thresholds.copy()
        inf_segment_speed_thresholds.append(10000)
        for i in range(len(inf_segment_speed_thresholds) - 1):
            values.append(
                len(df_segment_speeds.loc[
                        (df_segment_speeds['average_speed'] >= inf_segment_speed_thresholds[i]) &
                        (df_segment_speeds['average_speed'] < inf_segment_speed_thresholds[i + 1])]))
        return values

    def add_vehicle_speed_values(self, df, values):
        inf_segment_speed_thresholds = segment_speed_thresholds.copy()
        inf_segment_speed_thresholds.append(10000)
        df_vehicle_speeds = df.groupby('vehicle_id').agg({'weighted_speed': 'sum', 'driving_time': 'sum'})

        df_vehicle_speeds['average_speed'] = df_vehicle_speeds['weighted_speed'] / df_vehicle_speeds['driving_time']

        for i in range(len(inf_segment_speed_thresholds) - 1):
            values.append(
                len(df_vehicle_speeds.loc[
                        (df_vehicle_speeds['average_speed'] >= inf_segment_speed_thresholds[i]) &
                        (df_vehicle_speeds['average_speed'] < inf_segment_speed_thresholds[i + 1])]))
        return values

    def create_log(self, path: str):
        df_for_interval = self.get_df_for_next_interval()
        current_df = df_for_interval()
        total_df = pd.DataFrame(columns=current_df.columns)
        with open(path, 'w') as file:
            file.write(';'.join(self.get_columns()) + '\n')
            part_number = 1
            while current_df is not None:
                total_df = pd.concat([total_df, current_df])
                values = []

                # time offset
                values.append(self.time_interval_minutes * part_number)

                # number of active vehicles in time interval
                values.append(current_df['vehicle_id'].nunique())

                # number of active vehicles since start
                values.append(total_df['vehicle_id'].nunique())

                # number of vehicles with alternative selection
                current_alternatives, current_route_selection = self.get_vehicle_types_counts(current_df)

                for alternative in VehicleAlternatives.__members__.values():
                    values.append(current_alternatives[alternative])

                # number of vehicles with route selection
                for selection in VehicleRouteSelection.__members__.values():
                    values.append(current_route_selection[selection])

                # number of vehicles that finished journey in time interval
                values.append(current_df.loc[current_df['active'] == False, 'vehicle_id'].nunique())

                # number of vehicles that finished journey since start
                values.append(total_df.loc[total_df['active'] == False, 'vehicle_id'].nunique())

                # meters driven in time interval
                values.append(round(current_df['meters_driven'].sum()))

                # meters driven since start
                values.append(round(total_df['meters_driven'].sum()))

                # driving time in time interval
                current_df = self.calculate_driving_time(current_df)
                values.append(int(round(current_df['driving_time'].sum() / 60)))
                # set driving time to 1 for vehicles with zero driving time
                # this is done in order to calculate average speed correctly
                current_df.loc[current_df['driving_time'] == 0, 'driving_time'] = 1

                # driving time since start
                total_df_temp = self.calculate_driving_time(total_df)
                values.append(int(round(total_df_temp['driving_time'].sum() / 60)))
                total_df_temp.loc[total_df_temp['driving_time'] == 0, 'driving_time'] = 1

                # number of segments visited in time interval
                values.append(current_df['segment_id'].nunique())

                # number of segments visited since start
                values.append(total_df['segment_id'].nunique())

                # average speed in time interval
                current_df['weighted_speed'] = current_df['speed_mps'] * current_df['driving_time']
                current_avg_speed = current_df['weighted_speed'].sum() / current_df['driving_time'].sum()
                values.append(str(round(current_avg_speed, 2)).replace('.', ','))

                # average speed since start
                total_df_temp['weighted_speed'] = total_df_temp['speed_mps'] * total_df_temp['driving_time']
                total_avg_speed = total_df_temp['weighted_speed'].sum() / total_df_temp['driving_time'].sum()
                values.append(str(round(total_avg_speed, 2)).replace('.', ','))

                # number of segments with average speed in range in time interval
                values = self.add_segment_speed_values(current_df, values)

                # number of segments with average speed in range since start
                values = self.add_segment_speed_values(total_df_temp, values)

                # number of vehicles with average speed in range in time interval
                values = self.add_vehicle_speed_values(current_df, values)

                # number of vehicles with average speed in range since start
                values = self.add_vehicle_speed_values(total_df_temp, values)

                # add values to the csv
                file.write(";".join([str(value) for value in values]) + "\n")

                current_df = df_for_interval()
                part_number += 1


def create_simulation_log(simulation: Simulation, output_path: str, time_interval_minutes: int):
    simulation_log = SimulationLog(simulation, time_interval_minutes)
    simulation_log.create_log(output_path)


def create_simulations_comparison(simulation_paths: list[str], output_dir: str, time_interval_minutes: int):
    for path in simulation_paths:
        simulation = Simulation.load(path)
        output_csv_path = os.path.join(output_dir, Path(path).stem + ".csv")
        create_simulation_log(simulation, output_csv_path, time_interval_minutes)
