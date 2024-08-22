import pytest

import pytest
from unittest.mock import MagicMock
from datetime import datetime, timedelta

from ruth.data.map import Map
from ruth.data.segment import LengthMeters, Segment, SpeedKph, speed_kph_to_mps, SpeedMps
from ruth.globalview import GlobalView
from ruth.simulator.route import move_on_segment
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
        fcd_sampling_period=timedelta(seconds=0),
        status=""
    )
    return vehicle


@pytest.fixture
def setup_driving_route():
    segment1 = Segment(0, 1, 1000, SpeedKph(50.0))
    segment2 = Segment(1, 2, 1000, SpeedKph(50.0))
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


def test_vehicle_moves_normally(setup_vehicle, setup_driving_route, mock_gv_db, mock_routing_map, current_time):
    mock_gv_db.level_of_service_in_front_of_vehicle.return_value = 1.0
    mock_routing_map.get_current_max_speed = MagicMock(return_value=SpeedKph(50.0))

    vehicle_frequency = setup_vehicle.frequency

    result_time, result_position, result_speed = move_on_segment(
        vehicle=setup_vehicle,
        driving_route_part=setup_driving_route,
        current_time=current_time,
        gv_db=mock_gv_db,
        routing_map=mock_routing_map
    )

    expected_position = LengthMeters(speed_kph_to_mps(SpeedKph(50.0)) * 10)  # Speed * Time (10 seconds)
    assert result_position.position == expected_position
    assert result_speed == SpeedMps(speed_kph_to_mps(SpeedKph(50.0)))
    assert result_time == current_time + vehicle_frequency
    assert result_position.index == 0


def test_vehicle_finishes_segment(setup_vehicle, setup_driving_route, mock_gv_db, mock_routing_map, current_time):
    mock_gv_db.level_of_service_in_front_of_vehicle.return_value = 1.0
    mock_routing_map.get_current_max_speed = MagicMock(return_value=SpeedKph(50.0))
    setup_vehicle.start_distance_offset = setup_driving_route[0].length - LengthMeters(10.0)

    expected_travel_time = 10.0 / speed_kph_to_mps(SpeedKph(50.0))
    expected_position = LengthMeters(setup_driving_route[0].length)

    result_time, result_position, result_speed = move_on_segment(
        vehicle=setup_vehicle,
        driving_route_part=setup_driving_route,
        current_time=current_time,
        gv_db=mock_gv_db,
        routing_map=mock_routing_map
    )

    assert result_time == current_time + timedelta(seconds=expected_travel_time)
    assert result_position.position == expected_position
    assert result_position.index == 0
    assert result_speed == SpeedMps(speed_kph_to_mps(SpeedKph(50.0)))


def test_vehicle_moves_to_next_segment(setup_vehicle, setup_driving_route, mock_gv_db,
                                       mock_routing_map, current_time):
    mock_gv_db.level_of_service_in_front_of_vehicle.return_value = 1.0
    mock_routing_map.get_current_max_speed = MagicMock(return_value=SpeedKph(50.0))
    setup_vehicle.start_distance_offset = setup_driving_route[0].length

    result_time, result_position, result_speed = move_on_segment(
        vehicle=setup_vehicle,
        driving_route_part=setup_driving_route,
        current_time=current_time,
        gv_db=mock_gv_db,
        routing_map=mock_routing_map
    )

    expected_position = LengthMeters(speed_kph_to_mps(SpeedKph(50.0)) * 10)  # Speed * Time (10 seconds)
    assert result_time == current_time + setup_vehicle.frequency
    assert result_position.index == 1
    assert result_position.position == expected_position
    assert result_speed == SpeedMps(speed_kph_to_mps(SpeedKph(50.0)))


def test_vehicle_cannot_move_to_next_segment(setup_vehicle, setup_driving_route, mock_gv_db,
                                             mock_routing_map, current_time):
    mock_gv_db.level_of_service_in_front_of_vehicle.return_value = 1.0
    mock_routing_map.get_current_max_speed = MagicMock(return_value=SpeedKph(0.0))
    setup_vehicle.start_distance_offset = setup_driving_route[0].length

    result_time, result_position, result_speed = move_on_segment(
        vehicle=setup_vehicle,
        driving_route_part=setup_driving_route,
        current_time=current_time,
        gv_db=mock_gv_db,
        routing_map=mock_routing_map
    )

    expected_position = LengthMeters(setup_driving_route[0].length)
    assert result_time == current_time + setup_vehicle.frequency
    assert result_position.index == 0
    assert result_position.position == expected_position
    assert result_speed == SpeedMps(0.0)


def test_vehicle_stuck_in_traffic(setup_vehicle, setup_driving_route, mock_gv_db, mock_routing_map, current_time):
    mock_gv_db.level_of_service_in_front_of_vehicle.return_value = float("inf")
    mock_routing_map.get_current_max_speed = MagicMock(return_value=SpeedKph(50.0))

    previous_position = setup_vehicle.segment_position
    vehicle_frequency = setup_vehicle.frequency

    result_time, result_position, result_speed = move_on_segment(
        vehicle=setup_vehicle,
        driving_route_part=setup_driving_route,
        current_time=current_time,
        gv_db=mock_gv_db,
        routing_map=mock_routing_map
    )

    assert result_time == current_time + vehicle_frequency
    assert result_position == previous_position
    assert result_speed == SpeedMps(0.0)


def test_vehicle_is_at_the_end_of_route(setup_vehicle, setup_driving_route, mock_gv_db, mock_routing_map, current_time):
    """
    This scenario should not happen.
    """
    mock_gv_db.level_of_service_in_front_of_vehicle.return_value = 1.0
    mock_routing_map.get_current_max_speed = MagicMock(return_value=SpeedKph(50.0))
    setup_vehicle.start_index = 1
    setup_vehicle.start_distance_offset = setup_driving_route[1].length

    vehicle_frequency = setup_vehicle.frequency

    # expect index error because the vehicle is at the end of the route
    with pytest.raises(AssertionError):
        result_time, result_position, result_speed = move_on_segment(
            vehicle=setup_vehicle,
            driving_route_part=setup_driving_route[1:],
            current_time=current_time,
            gv_db=mock_gv_db,
            routing_map=mock_routing_map
        )

