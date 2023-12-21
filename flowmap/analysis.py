import os
from pathlib import Path
from ruth.simulator import Simulation
from flowmap.input import prepare_dataframe


def sort_df_by_timestamp(df):
    return df.sort_values(by=['timestamp'])


class SimulationLog:
    def __init__(self, simulation: Simulation):
        self.simulation = simulation
        self.df = prepare_dataframe(simulation.history.to_dataframe(), 1)
        self.df = sort_df_by_timestamp(self.df)

    def get_df_for_next_interval(self, time_interval_minutes: int):
        current_time = 0

        def create_df_for_interval():
            nonlocal current_time
            current_time += time_interval_minutes
            return current_time
        return create_df_for_interval

    def create_log(self, path: str, time_interval_minutes: int):
        pass


def create_simulation_log(simulation: Simulation, output_path: str, time_interval_minutes: int):
    simulation_log = SimulationLog(simulation)
    simulation_log.create_log(output_path, time_interval_minutes)


def create_simulations_comparison(simulation_paths: list[str], output_dir: str, time_interval_minutes: int):
    for path in simulation_paths:
        simulation = Simulation.load(path)
        output_csv_path = os.path.join(output_dir, Path(path).stem + ".csv")
        create_simulation_log(simulation, output_csv_path, time_interval_minutes)
