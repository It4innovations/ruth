import numpy as np

from dataclasses import dataclass, InitVar, field
from pandas import to_datetime
from networkx import MultiDiGraph as Graph
from typing import NewType

from ruth.metaclasses import Singleton

NodeId = NewType('NodeId', int)


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
    length: InitVar[float]
    node_from: NodeInTime = field(init=False)
    node_to: NodeInTime = field(init=False)

    node_from_: InitVar[NodeId]
    node_to_: InitVar[NodeId]
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

    def add_vehicle(self, start_offset_m, speed_mps):
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
    vehicle_id: int
    segment_id: str
    segment_length: float
    start_offset_m: float
    speed_mps: float
    status: str
    node_from: NodeId
    node_to: NodeId
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


def add_vehicle(record, divide: int, timestamp=None, start_offset_m=None, speed_mps=None):
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


def dataframe_to_sorted_records(df, graph, speed, fps):
    interval = speed / fps

    # change datetime to int
    df['timestamp'] = to_datetime(df['timestamp']).astype(np.int64) // 10 ** 6  # resolution in milliseconds

    df['timestamp'] = df['timestamp'].div(1000 * interval).round().astype(np.int64)
    df = df.groupby(['timestamp', 'vehicle_id']).first().reset_index()

    records = [Record(**kwargs, graph=graph) for kwargs in df.to_dict(orient='records')]
    records = sorted(records, key=lambda x: (x.vehicle_id, x.timestamp))

    return records


def fill_missing_times(df, graph, speed=1, fps=25, divide: int = 2):
    assert (
            divide >= 2
    ), f"Invalid value of divide '{divide}'. It must be greater or equal to 2."

    timed_segments = set()
    segments = set()

    records = dataframe_to_sorted_records(df, graph, speed, fps)

    for i, (processing_record, next_record) in enumerate(
            zip(records[:], records[1:] + [None])
    ):
        segments.add((processing_record.node_from, processing_record.node_to))
        if (
                next_record is None
                or processing_record.vehicle_id != next_record.vehicle_id
        ):
            timed_segments.add(add_vehicle(processing_record, divide))
        else:  # fill missing records
            new_timestamps = [*range(processing_record.timestamp, next_record.timestamp)]
            new_params = []

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

    return timed_segments, segments
