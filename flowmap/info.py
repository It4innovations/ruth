import networkx as nx
import pandas as pd
import itertools as itertools
from dataclasses import dataclass, field
from operator import attrgetter
from collections import namedtuple

from ruth.simulator import Simulation
from flowmap.input import dataframe_to_sorted_records, Record
from flowmap.time_unit import TimeUnit


def get_real_time(simulation, time_unit: TimeUnit):
    return (simulation.history.data[-1][0] - simulation.history.data[0][0]) / time_unit.value


def get_percentage(value, total):
    return round(value / total * 100, 2)


def print_in_columns(*args):
    print('{0:<100} {1:>10} {2:>10}'.format(*args))


@dataclass
class SimulationInfo:
    simulation: Simulation
    graph: nx.MultiDiGraph = field(init=False)
    records_df: pd.DataFrame = field(init=False)
    records: list[Record] = field(init=False)
    vehicles_count: int = field(init=False)
    min_timestamp: int = field(init=False)

    def __post_init__(self):
        self.graph = self.simulation.routing_map.network
        self.records_df = self.simulation.history.to_dataframe()
        self.records = dataframe_to_sorted_records(self.records_df, self.graph, 1, 1)
        self.vehicles_count = self.records_df['vehicle_id'].nunique()
        self.min_timestamp = min(self.records, key=attrgetter('timestamp')).timestamp

    def _get_vehicle_info(self, vehicle_records, last_timestamp_to_be_noted):
        segments_visited = set()
        finished_journey = True
        for record in vehicle_records:
            if record.timestamp > last_timestamp_to_be_noted:
                finished_journey = False
                break
            segments_visited.add(record.segment_id)
        return len(list(segments_visited)), finished_journey

    def _get_records_split_by_vehicle(self):
        vehicles_records = []
        for key, group in itertools.groupby(self.records, key=lambda record: record.vehicle_id):
            vehicles_records.append(list(group))
        return vehicles_records

    def print_info(self, minute):
        vehicles_records = self._get_records_split_by_vehicle()

        last_timestamp_to_be_noted = self.min_timestamp + 60 * minute
        segments_visited_counts = []
        segments_visited_counts_not_finished = []
        vehicles_finished_journey_count = 0

        for _, vehicle_records in enumerate(vehicles_records):
            segments_visited_count, finished_journey = self._get_vehicle_info(vehicle_records,
                                                                              last_timestamp_to_be_noted)
            segments_visited_counts.append(segments_visited_count)
            if finished_journey:
                vehicles_finished_journey_count += 1
            else:
                segments_visited_counts_not_finished.append(segments_visited_count)

        real_time_minutes = get_real_time(self.simulation, TimeUnit.MINUTES)
        print(f'\nINFO ABOUT MINUTE: {minute} ({get_percentage(minute, real_time_minutes)} % of simulation time)\n')
        print(f'Total number of vehicles: {self.vehicles_count}')

        vehicles_started_count = sum(c > 0 for c in segments_visited_counts)
        print_in_columns(
            'Number of vehicles that have started their journey:',
            vehicles_started_count,
            f'({get_percentage(vehicles_started_count, self.vehicles_count)} % of all vehicles)'
        )
        vehicles_left_initial_segment_count = sum(c > 1 for c in segments_visited_counts)
        print_in_columns(
            'Number of vehicles that have left their initial segment:',
            vehicles_left_initial_segment_count,
            f'({get_percentage(vehicles_left_initial_segment_count, self.vehicles_count)} % of all vehicles)'
        )
        print_in_columns(
            'Number of vehicles that have finished their journey:',
            vehicles_finished_journey_count,
            f'({get_percentage(vehicles_finished_journey_count, self.vehicles_count)} % of all vehicles)'
        )
        print_in_columns(
            'Average number of segments visited:',
            round(sum(segments_visited_counts) / self.vehicles_count, 2),
            ''
        )
        vehicles_not_finished_count = len(segments_visited_counts_not_finished)
        if vehicles_not_finished_count > 0:
            print_in_columns(
                'Average number of segments visited (only for vehicles that have not finished their journey):',
                round(sum(segments_visited_counts_not_finished) / vehicles_not_finished_count, 2),
                ''
            )

    def _check_completion_point(self, completion_point):
        if completion_point < 0:
            print('Status-at-point must be greater than or equal to 0. Setting it to 0.')
            completion_point = 0

        if completion_point > 1:
            print('Status-at-point must be less than or equal to 1. Setting it to 1.')
            completion_point = 1

        return completion_point

    def print_status_at_point(self, completion_point):
        completion_point = self._check_completion_point(completion_point)

        finished_vehicle_count_at_point = completion_point * self.vehicles_count
        current_finished_vehicle_count = 0
        timestamp_at_point = self.min_timestamp

        VehicleStatus = namedtuple('VehicleStatus', ['start_timestamp', 'end_timestamp', 'segments'])
        vehicles_statuses = {}

        vehicles_records = self._get_records_split_by_vehicle()
        for _, vehicle_records in enumerate(vehicles_records):
            vehicles_statuses[vehicle_records[0].vehicle_id] = VehicleStatus(vehicle_records[0].timestamp,
                                                                             vehicle_records[-1].timestamp, set())

        records_sorted = sorted(self.records, key=lambda x: x.timestamp)
        segments_count_finished_sum = 0
        for _, record in enumerate(records_sorted):
            if current_finished_vehicle_count >= finished_vehicle_count_at_point:
                break
            vehicle_status = vehicles_statuses[record.vehicle_id]
            vehicle_status.segments.add(record.segment_id)
            timestamp_at_point = record.timestamp
            if record.timestamp == vehicle_status.end_timestamp:
                current_finished_vehicle_count += 1
                segments_count_finished_sum += len(list(vehicle_status.segments))

        real_time_minutes = get_real_time(self.simulation, TimeUnit.MINUTES)
        remaining_vehicles_count = self.vehicles_count - current_finished_vehicle_count

        print(f'\nSTATUS AT STATE OF COMPLETION: {completion_point * 100} % '
              f'({current_finished_vehicle_count} vehicles finished their journey, '
              f'{remaining_vehicles_count} remaining)\n')
        print(f'Total number of vehicles: {self.vehicles_count}')

        minute = round((timestamp_at_point - self.min_timestamp) / 60)
        print(f'Minute: {minute} ({get_percentage(minute, real_time_minutes)} % of simulation time)')

        vehicles_started_count = 0
        vehicles_left_initial_segment_count = 0
        segments_count_sum = 0
        for vehicle_status in vehicles_statuses.values():
            segments = list(vehicle_status.segments)
            segments_count = len(segments)
            segments_count_sum += segments_count
            if segments_count > 0:
                vehicles_started_count += 1
            if segments_count > 1:
                vehicles_left_initial_segment_count += 1

        print_in_columns(
            'Number of vehicles that have started their journey:',
            vehicles_started_count,
            f'({get_percentage(vehicles_started_count, self.vehicles_count)} % of all vehicles)'
        )
        print_in_columns(
            'Number of vehicles that have left their initial segment:',
            vehicles_left_initial_segment_count,
            f'({get_percentage(vehicles_left_initial_segment_count, self.vehicles_count)} % of all vehicles)'
        )
        print_in_columns(
            'Average number of segments visited:',
            round(segments_count_sum / self.vehicles_count, 2),
            ''
        )
        vehicles_not_finished_count = self.vehicles_count - current_finished_vehicle_count
        if vehicles_not_finished_count > 0:
            print_in_columns(
                'Average number of segments visited (only for vehicles that have not finished their journey):',
                round((segments_count_sum - segments_count_finished_sum) / vehicles_not_finished_count, 2),
                ''
            )
