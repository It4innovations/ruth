from unittest.mock import MagicMock, patch, mock_open

import numpy as np
import pytest
from datetime import datetime
import csv

from ruth.data.map import BBox, Map, TemporarySpeed
from ruth.data.segment import SpeedKph


@pytest.fixture(scope='module')
def routing_map():
    bbox = BBox(50.16568920000002, 14.321441000000016, 50.020240399999985, 14.592499399999983)
    routing_map = Map(bbox=bbox, download_date='2024-01-10T00:00:00', with_speeds=True)
    return routing_map


@pytest.fixture
def setup_segment():
    segment1 = MagicMock()
    segment1.node_from = 25664661
    segment1.node_to = 27349583
    segment1.osmid = [863800707, 863800710, 19979079, 863800724, 545290749]
    segment1.highway = 'secondary'
    segment1.maxspeed = '50'
    segment1.length = 753
    segment1.speed_kph = 50.0
    segment1.routing_id = 3322
    segment1.current_speed = 50.0
    segment1.current_travel_time = 54.216

    segment2 = MagicMock()
    segment2.node_from = 27349583
    segment2.node_to = 27350859
    segment2.osmid = 545290748
    segment2.highway = 'secondary'
    segment2.maxspeed = '50'
    segment2.length = 133
    segment2.speed_kph = 50.0
    segment2.routing_id = 8554
    segment2.current_speed = 50.0
    segment2.current_travel_time = 9.576

    return [segment1, segment2]


def test_init_temporary_max_speeds(routing_map, setup_segment):
    csv_content = f"""node_from;node_to;speed;timestamp_from;timestamp_to
    {setup_segment[0].node_from};{setup_segment[0].node_to};0;2023-07-30 00:00:00;2023-07-30 00:05:00;
    {setup_segment[1].node_from};{setup_segment[1].node_to};60;2023-07-30 00:00:00;2023-07-30 00:05:00
    """

    speeds_path = "dummy_path"
    assert routing_map.temporary_speeds == []

    with patch("builtins.open", mock_open(read_data=csv_content)), patch("csv.reader", return_value=csv.reader(
            csv_content.splitlines(), delimiter=';')):
        # TEST
        routing_map.init_temporary_max_speeds(speeds_path)

    expected_temporary_speeds = [
        TemporarySpeed(setup_segment[0].node_from, setup_segment[0].node_to,
                       temporary_speed=SpeedKph(0),
                       original_max_speed=setup_segment[0].speed_kph,
                       timestamp_from=datetime(2023, 7, 30, 0, 0),
                       timestamp_to=datetime(2023, 7, 30, 0, 5), active=False),
        TemporarySpeed(setup_segment[1].node_from, setup_segment[1].node_to,
                       temporary_speed=SpeedKph(60),
                       original_max_speed=setup_segment[1].speed_kph,
                       timestamp_from=datetime(2023, 7, 30, 0, 0),
                       timestamp_to=datetime(2023, 7, 30, 0, 5), active=False)
    ]

    assert len(routing_map.temporary_speeds) == 2
    assert routing_map.temporary_speeds == expected_temporary_speeds


def test_update_temporary_max_speeds(routing_map, setup_segment):
    temporary_speeds = [
        TemporarySpeed(setup_segment[0].node_from, setup_segment[0].node_to,
                       temporary_speed=SpeedKph(0),
                       original_max_speed=setup_segment[0].speed_kph,
                       timestamp_from=datetime(2023, 7, 30, 0, 0),
                       timestamp_to=datetime(2023, 7, 30, 0, 5), active=False),
        TemporarySpeed(setup_segment[1].node_from, setup_segment[1].node_to,
                       temporary_speed=SpeedKph(60),
                       original_max_speed=setup_segment[1].speed_kph,
                       timestamp_from=datetime(2023, 7, 30, 0, 5),
                       timestamp_to=datetime(2023, 7, 30, 0, 10), active=False)
    ]
    routing_map.temporary_speeds = temporary_speeds.copy()

    # First update - 0:03
    timestamp = datetime(2023, 7, 30, 0, 3)
    new_current_speeds = routing_map.update_temporary_max_speeds(timestamp)

    assert len(routing_map.temporary_speeds) == 2
    # Current speed on first segment updated to lower value
    assert new_current_speeds == {
        (setup_segment[0].node_from, setup_segment[0].node_to): SpeedKph(0),
    }
    assert routing_map.temporary_speeds[0].active
    assert routing_map.temporary_speeds[1].active is False

    data1 = routing_map.current_network.edges[setup_segment[0].node_from, setup_segment[0].node_to]
    data2 = routing_map.current_network.edges[setup_segment[1].node_from, setup_segment[1].node_to]

    assert data1['speed_kph'] == 0
    assert data2['speed_kph'] == setup_segment[1].speed_kph
    assert data1['current_speed'] == 0
    assert data2['current_speed'] == setup_segment[1].current_speed
    assert data1['current_travel_time'] == np.inf
    assert data2['current_travel_time'] == setup_segment[1].current_travel_time

    # Second update - 0:06
    timestamp = datetime(2023, 7, 30, 0, 7)
    new_current_speeds = routing_map.update_temporary_max_speeds(timestamp)

    assert len(routing_map.temporary_speeds) == 1
    # Current speed on segment cannot be updated to higher value
    # On first segment from 0 to 50, on second segment from 50 to 60
    assert new_current_speeds == {}

    assert routing_map.current_network.edges[setup_segment[0].node_from, setup_segment[0].node_to]['speed_kph'] == setup_segment[0].speed_kph
    assert routing_map.current_network.edges[setup_segment[1].node_from, setup_segment[1].node_to]['speed_kph'] == 60


def test_update_temporary_max_speeds_no_temporary_speeds(routing_map, setup_segment):
    routing_map.temporary_speeds = []

    timestamp = datetime(2023, 7, 30, 0, 3)
    new_current_speeds = routing_map.update_temporary_max_speeds(timestamp)

    assert len(routing_map.temporary_speeds) == 0
    assert new_current_speeds == {}
    assert routing_map.current_network.edges[setup_segment[0].node_from, setup_segment[0].node_to]['speed_kph'] == setup_segment[0].speed_kph
    assert routing_map.current_network.edges[setup_segment[1].node_from, setup_segment[1].node_to]['speed_kph'] == setup_segment[1].speed_kph