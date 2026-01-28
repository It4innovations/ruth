import pytest
from unittest.mock import MagicMock
from datetime import datetime, timedelta

from ruth.data.map import Map
from ruth.data.segment import LengthMeters, Segment, SpeedKph, SegmentPosition, speed_kph_to_mps, SpeedMps
from ruth.globalview import GlobalView
from ruth.simulator.route import get_vehicle_speed
from ruth.vehicle import Vehicle


@pytest.fixture
def setup_vehicle():
    vehicle = Vehicle(
        id=0,
        time_offset=timedelta(seconds=0),
        frequency=timedelta(seconds=10),
        start_index=0,
        start_distance_offset=LengthMeters(0.0),
        origin_node=0,
        dest_node=2,
        osm_route=[0, 1, 2],
        active=True,
        fcd_sampling_period=timedelta(seconds=5),
        status=""
    )
    return vehicle


@pytest.fixture
def setup_driving_route():
    segment1 = Segment(0, 1, LengthMeters(1000.0), SpeedKph(50.0), 1)
    segment2 = Segment(1, 2, LengthMeters(1000.0), SpeedKph(50.0), 1)
    return [segment1, segment2]


@pytest.fixture
def mock_gv_db():
    gv_db = MagicMock(GlobalView)
    gv_db.level_of_service_in_front_of_vehicle = MagicMock()
    return gv_db


@pytest.fixture
def mock_routing_map():
    routing_map = MagicMock(Map)
    return routing_map


@pytest.fixture
def current_time():
    return datetime(2021, 1, 1, 0, 0, 0)


def test_get_vehicle_speed_normal_los(setup_vehicle, setup_driving_route, mock_gv_db, mock_routing_map, current_time):
    """Test getting vehicle speed with normal level of service."""
    mock_gv_db.level_of_service_in_front_of_vehicle.return_value = 1.0

    speed_mps, changed_segment = get_vehicle_speed(
        setup_vehicle, setup_driving_route, current_time, mock_gv_db, mock_routing_map, timedelta(seconds=0)
    )

    expected_speed = speed_kph_to_mps(SpeedKph(50.0))
    assert speed_mps == SpeedMps(expected_speed)
    assert changed_segment is False


def test_get_vehicle_speed_reduced_los(setup_vehicle, setup_driving_route, mock_gv_db, mock_routing_map, current_time):
    """Test getting vehicle speed with reduced level of service."""
    mock_gv_db.level_of_service_in_front_of_vehicle.return_value = 0.5

    speed_mps, changed_segment = get_vehicle_speed(
        setup_vehicle, setup_driving_route, current_time, mock_gv_db, mock_routing_map, timedelta(seconds=0)
    )

    expected_speed = speed_kph_to_mps(SpeedKph(25.0))  # 50 * 0.5
    assert speed_mps == SpeedMps(expected_speed)
    assert changed_segment is False


def test_get_vehicle_speed_zero_los(setup_vehicle, setup_driving_route, mock_gv_db, mock_routing_map, current_time):
    """Test getting vehicle speed when stuck in traffic jam (infinite LOS)."""
    mock_gv_db.level_of_service_in_front_of_vehicle.return_value = float("inf")

    speed_mps, changed_segment = get_vehicle_speed(
        setup_vehicle, setup_driving_route, current_time, mock_gv_db, mock_routing_map, timedelta(seconds=0)
    )

    assert speed_mps == SpeedMps(0.0)
    assert changed_segment is False


def test_get_vehicle_speed_at_segment_end(setup_vehicle, setup_driving_route, mock_gv_db, mock_routing_map, current_time):
    """Test when vehicle is at the end of segment and needs to move to next."""
    setup_vehicle.set_position(SegmentPosition(index=0, position=LengthMeters(1000.0)))
    mock_gv_db.level_of_service_in_front_of_vehicle.return_value = 1.0
    mock_routing_map.has_next_segment_closed = MagicMock(return_value=False)

    speed_mps, changed_segment = get_vehicle_speed(
        setup_vehicle, setup_driving_route, current_time, mock_gv_db, mock_routing_map, timedelta(seconds=0)
    )

    expected_speed = speed_kph_to_mps(SpeedKph(50.0))
    assert speed_mps == SpeedMps(expected_speed)
    assert changed_segment is True


def test_get_vehicle_speed_at_segment_end_blocked(setup_vehicle, setup_driving_route, mock_gv_db, mock_routing_map, current_time):
    """Test when vehicle is at segment end but next segment is closed."""
    setup_vehicle.set_position(SegmentPosition(index=0, position=LengthMeters(1000.0)))
    setup_vehicle.has_next_segment_closed = MagicMock(return_value=True)

    speed_mps, changed_segment = get_vehicle_speed(
        setup_vehicle, setup_driving_route, current_time, mock_gv_db, mock_routing_map, timedelta(seconds=0)
    )

    assert speed_mps == SpeedMps(0.0)
    assert changed_segment is False
