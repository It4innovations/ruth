from unittest.mock import MagicMock, patch, mock_open

import numpy as np
import pytest
from datetime import datetime
import csv

from ruth.data.map import BBox, Map, TemporarySpeed
from ruth.data.segment import SpeedKph, SegmentId


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
    segment1.speed_kph = 50.0
    segment1.length = 753
    segment1.current_travel_time = segment1.length / (segment1.speed_kph / 3.6)

    segment2 = MagicMock()
    segment2.node_from = 27349583
    segment2.node_to = 27350859
    segment2.speed_kph = 50.0
    segment2.length = 133
    segment2.current_travel_time = segment2.length / (segment2.speed_kph / 3.6)

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
    """
    Test updates speeds
    Time 0:00 -> first segment speed is set to 0
    Time 0:05 -> first segment back to 50, second segment speed is set to 30
    """
    # check if initial speeds are set to original max speeds
    for segment in setup_segment:
        data = routing_map.current_network.edges[segment.node_from, segment.node_to]
        assert data['maxspeed'] == segment.speed_kph
        assert data['speed_kph'] == segment.speed_kph
        assert data['current_speed'] == segment.speed_kph
        assert data['current_travel_time'] == segment.current_travel_time

    temporary_speeds = [
        TemporarySpeed(setup_segment[0].node_from, setup_segment[0].node_to,
                       temporary_speed=SpeedKph(0),
                       original_max_speed=setup_segment[0].speed_kph,
                       timestamp_from=datetime(2023, 7, 30, 0, 0),
                       timestamp_to=datetime(2023, 7, 30, 0, 5), active=False),
        TemporarySpeed(setup_segment[1].node_from, setup_segment[1].node_to,
                       temporary_speed=SpeedKph(30),
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
    assert new_current_speeds == {(setup_segment[0].node_from, setup_segment[0].node_to): SpeedKph(0)}
    assert routing_map.temporary_speeds[0].active
    assert routing_map.temporary_speeds[1].active is False

    data1 = routing_map.current_network.edges[setup_segment[0].node_from, setup_segment[0].node_to]
    data2 = routing_map.current_network.edges[setup_segment[1].node_from, setup_segment[1].node_to]

    assert data1['speed_kph'] == 0
    assert data2['speed_kph'] == setup_segment[1].speed_kph
    assert data1['current_speed'] == 0
    assert data2['current_speed'] == setup_segment[1].speed_kph
    assert data1['current_travel_time'] == np.inf
    assert data2['current_travel_time'] == setup_segment[1].current_travel_time

    # Second update - 0:06
    timestamp = datetime(2023, 7, 30, 0, 7)
    new_current_speeds = routing_map.update_temporary_max_speeds(timestamp)

    assert len(routing_map.temporary_speeds) == 1
    assert routing_map.current_network.edges[setup_segment[0].node_from, setup_segment[0].node_to]['speed_kph'] == setup_segment[0].speed_kph
    assert routing_map.current_network.edges[setup_segment[1].node_from, setup_segment[1].node_to]['speed_kph'] == 30

    # Current speed on segment cannot be updated to higher value
    assert new_current_speeds == {
        (setup_segment[0].node_from, setup_segment[0].node_to): SpeedKph(0),
        (setup_segment[1].node_from, setup_segment[1].node_to): SpeedKph(30)
    }

    # Check get current max speed and get original max speed functions after the updates
    # First segment: current max speed should be restored to original, original should remain unchanged
    assert routing_map.get_current_max_speed(setup_segment[0].node_from, setup_segment[0].node_to) == setup_segment[0].speed_kph
    assert routing_map.get_original_max_speed(setup_segment[0].node_from, setup_segment[0].node_to) == setup_segment[0].speed_kph

    # Second segment: current max speed should be temporarily reduced, original should remain unchanged
    assert routing_map.get_current_max_speed(setup_segment[1].node_from, setup_segment[1].node_to) == 30
    assert routing_map.get_original_max_speed(setup_segment[1].node_from, setup_segment[1].node_to) == setup_segment[1].speed_kph


# ============================================================================
# ROUTE CLOSURE TESTS - Test detecting closed/blocked routes
# ============================================================================

def test_round_speed():
    """Test rounding speeds to nearest integer."""
    from ruth.data.map import round_speed
    assert round_speed(49.4) == 49
    assert round_speed(49.5) == 50
    assert round_speed(49.6) == 50
    assert round_speed(50.0) == 50
    assert round_speed(0.0) == 0
    assert round_speed(0.4) == 0
    assert round_speed(0.5) == 1
    assert round_speed(100.5) == 101


def test_get_osm_segment_id():
    """Test OSM segment ID generation."""
    from ruth.data.map import get_osm_segment_id
    segment_id = get_osm_segment_id(123, 456)
    assert segment_id == "OSM123T456"

    segment_id = get_osm_segment_id(0, 999999)
    assert segment_id == "OSM0T999999"


def test_osm_route_to_segment_ids():
    """Test converting single segment route."""
    from ruth.data.map import osm_route_to_segment_ids
    route = [1, 2]
    segment_ids = osm_route_to_segment_ids(route)
    assert segment_ids == ["OSM1T2"]

    route = [1, 2, 3, 4]
    segment_ids = osm_route_to_segment_ids(route)
    assert segment_ids == ["OSM1T2", "OSM2T3", "OSM3T4"]

# ============================================================================
# BBOX TESTS - Test BBox class
# ============================================================================

def test_bbox_initialization():
    """Test BBox creation with all coordinates."""
    bbox = BBox(north=50.5, west=10.2, south=50.0, east=10.8)
    assert bbox.north == 50.5
    assert bbox.west == 10.2
    assert bbox.south == 50.0
    assert bbox.east == 10.8


def test_bbox_get_coords():
    """Test BBox coordinate retrieval in correct order."""
    bbox = BBox(north=50.5, west=10.2, south=50.0, east=10.8)
    coords = bbox.get_coords()
    assert coords == (50.5, 10.2, 50.0, 10.8)


def test_bbox_name_property():
    """Test BBox name generation from coordinates."""
    bbox = BBox(north=50.5, west=10.2, south=50.0, east=10.8)
    name = bbox.name
    assert "50_5" in name
    assert "10_2" in name
    assert "50_0" in name
    assert "10_8" in name


def test_bbox_name_with_decimals():
    """Test BBox name properly replaces decimal points."""
    bbox = BBox(north=50.123, west=10.456, south=50.789, east=10.012)
    name = bbox.name
    assert "." not in name
    assert "50_123" in name

# ============================================================================
# SEGMENT OPERATIONS - Test segment retrieval and conversion
# ============================================================================

def test_get_osm_segment_caching(routing_map):
    """Test that segments are cached and same object is returned."""
    edges = list(routing_map.original_network.edges())
    if edges:
        node_from, node_to = edges[0]
        segment1 = routing_map.get_osm_segment(node_from, node_to)
        segment2 = routing_map.get_osm_segment(node_from, node_to)
        # check if cache size is 1
        assert len(routing_map._segment_cache) == 1
        # Same object should be returned from cache
        assert segment1 is segment2


def test_osm_route_to_py_segments(routing_map, setup_segment):
    """Test converting OSM route (node list) to segments."""
    route = [setup_segment[0].node_from, setup_segment[0].node_to, setup_segment[1].node_to]
    segments = routing_map.osm_route_to_py_segments(route)

    assert len(segments) == 2
    assert segments[0].node_from == setup_segment[0].node_from
    assert segments[0].node_to == setup_segment[0].node_to


# ============================================================================
# SPEED MANAGEMENT - Test speed initialization and updates
# ============================================================================

def test_init_current_speeds(routing_map):
    """Test initializing current speeds from network."""
    routing_map.init_current_speeds()

    # Check that current_speed and current_travel_time are set for all edges
    for u, v in routing_map.current_network.edges():
        assert 'current_speed' in routing_map.current_network[u][v]
        assert 'current_travel_time' in routing_map.current_network[u][v]

        # check if current_speed equals maxspeed and speed_kph
        data = routing_map.current_network.edges[u, v]
        assert data['current_speed'] == data['maxspeed']
        assert data['current_speed'] == data['speed_kph']
        # check if current_travel_time is correctly calculated
        expected_travel_time = data['length'] / (data['current_speed'] / 3.6)
        assert pytest.approx(data['current_travel_time'], abs=0.01) == expected_travel_time


def test_update_current_speeds_increment_map_id(routing_map, setup_segment):
    """Test that updating speeds increments map ID."""
    initial_id = routing_map.get_map_id()

    segment_id = SegmentId((setup_segment[0].node_from, setup_segment[0].node_to))
    segments_to_update = {segment_id: SpeedKph(30.0)}
    routing_map.update_current_speeds(segments_to_update)

    assert routing_map.get_map_id() == initial_id + 1


def test_update_current_speeds_caps_at_max(routing_map, setup_segment):
    """Test that current speed doesn't exceed max speed."""
    # Try to set speed higher than max
    segment_id = SegmentId((setup_segment[0].node_from, setup_segment[0].node_to))
    segments_to_update = {segment_id: SpeedKph(200.0)}
    new_speeds = routing_map.update_current_speeds(segments_to_update)

    # Should be capped at original max speed
    assert new_speeds[(setup_segment[0].node_from, setup_segment[0].node_to)] == setup_segment[0].speed_kph


def test_update_current_speeds_none_restores_max(routing_map, setup_segment):
    """Test that None speed restores original max speed."""
    # First reduce speed
    segment_id = SegmentId((setup_segment[0].node_from, setup_segment[0].node_to))
    routing_map.update_current_speeds({segment_id: SpeedKph(20.0)})

    # Then restore with None
    new_speeds = routing_map.update_current_speeds({segment_id: None})

    # Should restore to original max speed
    assert new_speeds[(setup_segment[0].node_from, setup_segment[0].node_to)] == setup_segment[0].speed_kph


# ============================================================================
# TRAVEL TIME CALCULATIONS - Test travel time computations
# ============================================================================


def test_get_path_travel_time(routing_map, setup_segment):
    """Test getting total travel time for a path."""
    path = [setup_segment[0].node_from, setup_segment[0].node_to, setup_segment[1].node_to]
    total_time = routing_map.get_path_travel_time(path)

    # Should be sum of individual segment times
    time_1 = routing_map.get_segment_travel_time(setup_segment[0].node_from, setup_segment[0].node_to)
    time_2 = routing_map.get_segment_travel_time(setup_segment[0].node_to, setup_segment[1].node_to)

    assert pytest.approx(total_time, abs=0.01) == time_1 + time_2


def test_check_if_travel_time_is_faster_true(routing_map, setup_segment):
    """Test path is faster than time limit (true case)."""
    path = [setup_segment[0].node_from, setup_segment[0].node_to]
    time_limit = 1000.0  # Very high limit

    is_faster = routing_map.check_if_travel_time_is_faster(path, time_limit)
    assert is_faster is True


def test_check_if_travel_time_is_faster_false(routing_map, setup_segment):
    """Test path is not faster than time limit (false case)."""
    path = [setup_segment[0].node_from, setup_segment[0].node_to, setup_segment[1].node_to]
    time_limit = 0.01  # Very low limit

    is_faster = routing_map.check_if_travel_time_is_faster(path, time_limit)
    assert is_faster is False


# ============================================================================
# ROUTE CLOSURE TESTS - Test detecting closed/blocked routes
# ============================================================================

def test_is_route_closed_false(routing_map, setup_segment):
    """Test route with all open segments returns False."""
    path = [setup_segment[0].node_from, setup_segment[0].node_to, setup_segment[1].node_to]
    is_closed = routing_map.is_route_closed(path)
    assert is_closed is False

def test_is_route_closed_true(routing_map, setup_segment):
    """Test route with a closed segment returns True."""
    # set new temporary speed to 0 for first segment, use function to update current speeds
    temp_speed = TemporarySpeed(
        node_from=setup_segment[0].node_from,
        node_to=setup_segment[0].node_to,
        temporary_speed=SpeedKph(0),
        original_max_speed=setup_segment[0].speed_kph,
        timestamp_from=datetime(2024, 1, 1, 0, 0, 0),
        timestamp_to=datetime(2024, 1, 1, 1, 0, 0),
        active=False
    )
    routing_map.temporary_speeds = [temp_speed]
    routing_map.update_temporary_max_speeds(datetime(2024, 1, 1, 0, 30, 0))

    path = [setup_segment[0].node_from, setup_segment[0].node_to, setup_segment[1].node_to]
    is_closed = routing_map.is_route_closed(path)
    assert is_closed is True


# ============================================================================
# TEMPORARY SPEEDS TESTS - Test temporary speed scheduling and updates
# ============================================================================

def test_has_temporary_speeds_planned_true(routing_map):
    """Test checking if temporary speeds are planned (true case)."""
    ts = TemporarySpeed(
        node_from=1,
        node_to=2,
        temporary_speed=SpeedKph(30.0),
        original_max_speed=SpeedKph(50.0),
        timestamp_from=datetime(2021, 1, 1, 10, 0, 0),
        timestamp_to=datetime(2021, 1, 1, 11, 0, 0),
        active=False
    )
    routing_map.temporary_speeds = [ts]

    assert routing_map.has_temporary_speeds_planned() is True


def test_has_temporary_speeds_planned_false(routing_map):
    """Test checking if temporary speeds are planned (false case)."""
    routing_map.temporary_speeds = []
    assert routing_map.has_temporary_speeds_planned() is False
