from datetime import timedelta
import numpy as np
from dataclasses import dataclass, InitVar, field
from collections import defaultdict
from pandas import to_datetime
from networkx import MultiDiGraph as Graph
from typing import NewType

from ruth.metaclasses import Singleton

NodeId = NewType('NodeId', int)


class PreprocessedData:
    def __init__(self,
                 segments: set,
                 timed_segments: set,
                 number_of_vehicles: int,
                 number_of_finished_vehicles_in_time: defaultdict,
                 total_simulation_time_in_time_s: defaultdict = defaultdict(lambda: timedelta(0)),
                 total_meters_driven_in_time: defaultdict = defaultdict(int)):
        self.segments = segments
        self.timed_segments = timed_segments
        self.number_of_vehicles = number_of_vehicles
        self.number_of_finished_vehicles_in_time = number_of_finished_vehicles_in_time
        self.total_simulation_time_in_time_s = total_simulation_time_in_time_s
        self.total_meters_driven_in_time = total_meters_driven_in_time


@dataclass
class NodeInTime(metaclass=Singleton):
    id: NodeId
    timestamp: int

    def __post_init__(self):
        self.vehicle_count = 0

    def add_vehicle(self):
        self.vehicle_count += 1


@dataclass
class SegmentInTime(metaclass=Singleton):
    node_from: NodeInTime = field(init=False)
    node_to: NodeInTime = field(init=False)

    node_from_: InitVar[NodeId]
    node_to_: InitVar[NodeId]
    length: InitVar[float]
    timestamp: InitVar[int]
    divide: InitVar[int] = field(default=2)

    def __post_init__(self, node_from_: NodeId, node_to_: NodeId, length: float, timestamp, divide):
        self.node_from = NodeInTime(node_from_, timestamp)
        self.node_to = NodeInTime(node_to_, timestamp)
        self.length = length
        self.inner_counts = [0] * (divide - 2)  # -2 for two nodes
        self.offsets = []
        self.divide = divide
        self.speeds_list = [[] for _ in range(divide)]

    def __hash__(self):
        return hash((self.node_from.id, self.node_to.id, self.timestamp))

    def add_vehicle(self, start_offset_m: float, speed_mps: float):
        self.offsets.append(start_offset_m)
        step = self.length / self.divide
        division = int(start_offset_m // step)
        if division >= self.divide:
            division = self.divide - 1

        self.speeds_list[division].append(speed_mps)

        if division == 0:
            self.node_from.add_vehicle()
        elif division >= self.divide - 1:
            self.node_to.add_vehicle()
        else:
            self.inner_counts[division - 1] += 1  # -1 for node_from

    @property
    def timestamp(self):
        return self.node_from.timestamp

    @property
    def counts(self):
        return [
            self.node_from.vehicle_count,
            *self.inner_counts,
            self.node_to.vehicle_count
        ]

    @property
    def speeds(self):
        return [sum(segment_part) / len(segment_part) if segment_part else -1 for segment_part in self.speeds_list]

    @property
    def cars_offsets(self):
        if sum(self.counts) > 5:
            return sum(self.offsets) / len(self.offsets)
        return None


@dataclass
class Record:
    timestamp: int
    timestamp_datetime: int
    vehicle_id: int
    segment_id: str
    segment_length: float
    start_offset_m: float
    speed_mps: float
    status: str
    node_from: NodeId
    node_to: NodeId
    meters_driven: float
    active: bool = None
    length: InitVar[float] = None
    graph: InitVar[Graph] = None

    def __post_init__(self, length, graph=None):
        self.length = length

        if (
                length is None
                and graph is not None
                and (self.node_from is not None or self.node_to is not None)
        ):
            data = graph.get_edge_data(self.node_from, self.node_to)
            assert data is not None

            data = data[0]
            assert 'length' in data
            self.length = data['length']


def add_vehicle(record: Record, divide: int, timestamp: int = None, start_offset_m: float = None, speed_mps: float = None):
    """Add vehicle to **singleton** segment in time element."""
    if timestamp is None:
        timestamp = record.timestamp
    if start_offset_m is None:
        start_offset_m = record.start_offset_m
    if speed_mps is None:
        speed_mps = record.speed_mps

    timed_seg = SegmentInTime(
        record.node_from, record.node_to, record.length, timestamp, divide
    )
    timed_seg.add_vehicle(start_offset_m, speed_mps)

    return timed_seg


def choose_one_row_per_vehicle_timestamp(df):
    # the goal is to filter the DataFrame to keep only one row per vehicle per timestamp
    # for the first timestamp of each vehicle, we want to keep the first row
    # for every other timestamp of each vehicle, we want to keep the last row
    # so that we don't change the origin and destination position and timestamp of each vehicle

    # identify the first occurrence of each vehicle in the sorted DataFrame
    first_occurrence_mask = ~df.duplicated(subset=['vehicle_id'])

    # identify the last occurrence of each vehicle and timestamp in the sorted DataFrame
    last_occurrence_mask = ~df.duplicated(subset=['vehicle_id', 'timestamp'], keep='last')

    # filter the DataFrame based on the two masks
    df_filtered = df[first_occurrence_mask | last_occurrence_mask]

    # this might still keep two rows for the first timestamp of each vehicle
    # so we need to remove duplicates and keep the first row from the group
    df_filtered = df_filtered.drop_duplicates(subset=['vehicle_id', 'timestamp'], keep='first')

    # sum the meters driven
    df_filtered['meters_driven'] = df.groupby(['vehicle_id', 'timestamp'])['meters_driven'].transform('sum')

    return df_filtered


def add_meters_driven_column(df):
    # add a column with the number of meters driven by each vehicle for simulation statistics
    # this has to be calculated before filtering the DataFrame since it might filter out some segments completely

    # sort the DataFrame by timestamp within each vehicle group
    df = df.sort_values(['vehicle_id', 'timestamp'])

    # calculate the number of meters driven from the previous timestamp
    df['previous_segment_ig'] = df['segment_id'].shift(1)
    df['previous_vehicle_id'] = df['vehicle_id'].shift(1)
    df['previous_start_offset_m'] = df['start_offset_m'].shift(1)
    df['previous_segment_length'] = df['segment_length'].shift(1)
    df['meters_driven'] = df['start_offset_m'] - df['previous_start_offset_m']
    df.loc[df['previous_segment_ig'] != df['segment_id'], 'meters_driven'] += df['previous_segment_length']

    # clear the meters driven for the first timestamp of each vehicle
    df.loc[df['previous_vehicle_id'] != df['vehicle_id'], 'meters_driven'] = 0

    # drop the columns that are not needed anymore
    df.drop(columns=['previous_segment_ig',
                     'previous_vehicle_id',
                     'previous_start_offset_m',
                     'previous_segment_length'],
            inplace=True)
    return df


def prepare_dataframe(df, interval):
    df['timestamp_datetime'] = df['timestamp']
    # change timestamp to the number of frame in the animation
    df['timestamp'] = to_datetime(df['timestamp']).astype(np.int64) // 10 ** 6  # resolution in milliseconds
    df['timestamp'] = df['timestamp'].div(1000 * interval).round().astype(np.int64)
    if 'segment_id' not in df.columns:
        df['segment_id'] = 'OSM' + df['node_from'].astype(str) + 'T' + df['node_to'].astype(str)

    df = add_meters_driven_column(df)
    df = choose_one_row_per_vehicle_timestamp(df)
    return df


def dataframe_to_sorted_records(df, graph, speed, fps):
    interval = speed / fps
    df = prepare_dataframe(df, interval)
    records = [Record(**kwargs, graph=graph) for kwargs in df.to_dict(orient='records')]
    records = sorted(records, key=lambda x: (x.vehicle_id, x.timestamp))

    return records


def get_number_of_vehicles(df):
    return df['vehicle_id'].nunique()


def is_last_record_for_vehicle(processing_record, next_record):
    return next_record is None or processing_record.vehicle_id != next_record.vehicle_id


def preprocess_data(df, graph, speed=1, fps=25, divide: int = 2):
    assert (
            divide >= 2
    ), f"Invalid value of divide '{divide}'. It must be greater or equal to 2."

    timed_segments = set()
    segments = set()
    number_of_vehicles = get_number_of_vehicles(df)
    number_of_finished_vehicles_in_time = defaultdict(int)
    total_simulation_time_in_time_s = defaultdict(lambda: timedelta(0))
    total_meters_driven_in_time = defaultdict(int)
    records = dataframe_to_sorted_records(df, graph, speed, fps)

    for i, (processing_record, next_record) in enumerate(
            zip(records[:], records[1:] + [None])
    ):
        segments.add((processing_record.node_from, processing_record.node_to))
        total_meters_driven_in_time[processing_record.timestamp] += processing_record.meters_driven
        if is_last_record_for_vehicle(processing_record, next_record):
            timed_segments.add(add_vehicle(processing_record, divide))
            if processing_record.active is False:
                number_of_finished_vehicles_in_time[processing_record.timestamp] += 1

        else:  # fill missing records
            new_timestamps = [*range(processing_record.timestamp, next_record.timestamp)]
            total_simulation_time_in_time_s[next_record.timestamp] += \
                next_record.timestamp_datetime - processing_record.timestamp_datetime
            new_speeds = np.linspace(
                processing_record.speed_mps,
                next_record.speed_mps,
                len(new_timestamps),
                endpoint=False
            )

            if processing_record.segment_id == next_record.segment_id:
                new_offsets = np.linspace(
                    processing_record.start_offset_m,
                    next_record.start_offset_m,
                    len(new_timestamps),
                    endpoint=False
                )
                processing_segments = [processing_record] * len(new_offsets)
            else:
                # 1. fill missing offset between two consecutive records
                new_offsets = np.linspace(
                    processing_record.start_offset_m,
                    next_record.start_offset_m + processing_record.length,
                    len(new_timestamps),
                    endpoint=False
                )
                # 2. adjust offset where the segment change
                processing_segments = np.where(
                    new_offsets > processing_record.length,
                    next_record,
                    processing_record
                )
                new_offsets = np.where(
                    new_offsets > processing_record.length,
                    new_offsets - processing_record.length,
                    new_offsets
                )

            new_params = zip(new_timestamps, new_offsets, new_speeds)

            for record, (timestamp, start_offset_m, speed_mps) in zip(processing_segments, new_params):
                timed_segments.add(add_vehicle(record, divide, timestamp, start_offset_m, speed_mps))

    min_timestamp = min(timed_segments, key=lambda x: x.timestamp).timestamp
    max_timestamp = max(timed_segments, key=lambda x: x.timestamp).timestamp
    for timestamp in range(min_timestamp + 1, max_timestamp + 1):
        total_meters_driven_in_time[timestamp] += total_meters_driven_in_time[timestamp - 1]
        total_simulation_time_in_time_s[timestamp] += total_simulation_time_in_time_s[timestamp - 1]
        number_of_finished_vehicles_in_time[timestamp] += number_of_finished_vehicles_in_time[timestamp - 1]

    return PreprocessedData(segments,
                            timed_segments,
                            number_of_vehicles,
                            number_of_finished_vehicles_in_time,
                            total_simulation_time_in_time_s,
                            total_meters_driven_in_time)
