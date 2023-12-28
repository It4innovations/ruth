import os
from pathlib import Path

import pandas as pd
from ruth.simulator import Simulation
from ruth.vehicle import VehicleAlternatives, VehicleRouteSelection
from flowmap.input import prepare_dataframe
from flowmap.flowmapframe.speeds import segment_speed_thresholds


def sort_df_by_timestamp(df):
    return df.sort_values(by=['timestamp'])


class Log:
    def __init__(self):
        pass


class SimulationLog:
    def __init__(self, simulation: Simulation):
        self.simulation = simulation
        self.df = prepare_dataframe(simulation.history.to_dataframe(), 1)
        self.df = sort_df_by_timestamp(self.df)
        self.vehicle_alternatives = {}
        self.vehicle_route_selection = {}

        for vehicle in simulation.vehicles:
            self.vehicle_alternatives[vehicle.id] = vehicle.alternatives
            self.vehicle_route_selection[vehicle.id] = vehicle.route_selection

    def get_df_for_next_interval(self, time_interval_minutes: int):
        current_timestamp = self.df['timestamp'][0]
        last_timestamp = self.df['timestamp'][-1]

        def create_df_for_interval():
            nonlocal current_timestamp
            if current_timestamp >= last_timestamp:
                return None
            next_timestamp = current_timestamp + time_interval_minutes
            current_df = self.df[(self.df['timestamp'] >= current_timestamp) & (self.df['timestamp'] < next_timestamp)]
            current_timestamp = next_timestamp
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
            'driving time in time interval',
            'driving time in total',
            'number of segments visited in time interval',
            'number of segments visited in total',
            'average speed in time interval',
            'average speed in total',
        ])

        inf_segment_speed_thresholds = segment_speed_thresholds.copy()
        inf_segment_speed_thresholds.append('inf')
        for i in range(len(inf_segment_speed_thresholds) - 1):
            columns.append(f"number of segments with average speed in range {inf_segment_speed_thresholds[i]} - {inf_segment_speed_thresholds[i + 1]} km/h in time interval")
            columns.append(f"number of segments with average speed in range {inf_segment_speed_thresholds[i]} - {inf_segment_speed_thresholds[i + 1]} km/h in total")
            columns.append(f"number of vehicles with average speed in range {inf_segment_speed_thresholds[i]} - {inf_segment_speed_thresholds[i + 1]} km/h in time interval")
            columns.append(f"number of vehicles with average speed in range {inf_segment_speed_thresholds[i]} - {inf_segment_speed_thresholds[i + 1]} km/h in total")
        return columns

    def create_log(self, path: str, time_interval_minutes: int):
        df_for_interval = self.get_df_for_next_interval(time_interval_minutes)
        current_df = df_for_interval()
        total_df = pd.DataFrame(columns=current_df.columns)
        with open(path, 'w') as file:
            file.write(','.join(self.get_columns()))
            i = 1
            while current_df is not None:
                total_df = total_df.append(current_df)
                values = []

                # time offset
                values.append(time_interval_minutes * i)

                # number of active vehicles in time interval
                values.append(current_df['vehicle'].nunique())

                # number of active vehicles in total
                values.append(total_df['vehicle'].nunique())

                # number of vehicles with alternative selection
                active_vehicles = current_df['vehicle'].unique()
                for alternative in VehicleAlternatives:
                    values.append(len([vehicle for vehicle in active_vehicles if
                                       self.vehicle_alternatives[vehicle] == alternative]))

                # number of vehicles with route selection
                for selection in VehicleRouteSelection:
                    values.append(len([vehicle for vehicle in active_vehicles if
                                       self.vehicle_route_selection[vehicle] == selection]))

                # number of vehicles that finished journey in time interval
                values.append(current_df.loc[current_df['active'] == False, 'vehicle'].nunique())

                # number of vehicles that finished journey in total
                values.append(total_df.loc[total_df['active'] == False, 'vehicle'].nunique())

                current_df = df_for_interval()




def create_simulation_log(simulation: Simulation, output_path: str, time_interval_minutes: int):
    simulation_log = SimulationLog(simulation)
    simulation_log.create_log(output_path, time_interval_minutes)


def create_simulations_comparison(simulation_paths: list[str], output_dir: str, time_interval_minutes: int):
    for path in simulation_paths:
        simulation = Simulation.load(path)
        output_csv_path = os.path.join(output_dir, Path(path).stem + ".csv")
        create_simulation_log(simulation, output_csv_path, time_interval_minutes)
