import pytest
from unittest.mock import MagicMock, call
from datetime import datetime, timedelta

from ruth.data.map import Map
from ruth.data.segment import LengthMeters, Segment, SpeedKph, SegmentPosition, speed_kph_to_mps, SpeedMps
from ruth.globalview import GlobalView
from ruth.simulator.route import advance_vehicles_with_queues
from ruth.simulator.queues import QueuesManager
from ruth.vehicle import Vehicle


@pytest.fixture
def setup_vehicles():
    """Create multiple test vehicles."""
    vehicles = [
        Vehicle(
            id=i,
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
        for i in range(3)
    ]
    return vehicles


@pytest.fixture
def setup_driving_route():
    segment1 = Segment(0, 1, LengthMeters(1000.0), SpeedKph(50.0), 1)
    segment2 = Segment(1, 2, LengthMeters(1000.0), SpeedKph(50.0), 1)
    return [segment1, segment2]


@pytest.fixture
def mock_gv_db():
    gv_db = MagicMock(GlobalView)
    gv_db.level_of_service_in_front_of_vehicle = MagicMock(return_value=1.0)
    return gv_db


@pytest.fixture
def mock_routing_map():
    routing_map = MagicMock(Map)
    routing_map.osm_route_to_py_segments = MagicMock(return_value=[
        Segment(0, 1, LengthMeters(1000.0), SpeedKph(50.0), 1),
        Segment(1, 2, LengthMeters(1000.0), SpeedKph(50.0), 1)
    ])
    return routing_map


@pytest.fixture
def mock_queues_manager():
    qm = MagicMock(QueuesManager)
    qm.queues = {(0, 1): set(), (1, 2): set()}
    qm.batch_update = MagicMock()
    return qm


@pytest.fixture
def current_time():
    return datetime(2021, 1, 1, 0, 0, 0)


def test_advance_vehicles_no_vehicles(mock_gv_db, mock_routing_map, mock_queues_manager, current_time):
    """Test with no vehicles to move."""
    fcds, vehicles_moved = advance_vehicles_with_queues(
        [], current_time, mock_gv_db, mock_routing_map, mock_queues_manager, timedelta(seconds=0)
    )

    assert fcds == []
    assert vehicles_moved is False
    mock_queues_manager.batch_update.assert_called_once()


def test_advance_vehicles_not_in_queues(setup_vehicles, mock_gv_db, mock_routing_map, mock_queues_manager, current_time):
    """Test vehicles that are not in any queues."""
    # Ensure vehicles are not in any queue
    mock_queues_manager.queues = {(0, 1): set(), (1, 2): set()}

    fcds, vehicles_moved = advance_vehicles_with_queues(
        setup_vehicles, current_time, mock_gv_db, mock_routing_map, mock_queues_manager, timedelta(seconds=0)
    )

    # Should have moved vehicles and generated FCD records
    assert len(fcds) > 0
    assert vehicles_moved is True
    mock_queues_manager.batch_update.assert_called_once()


def test_advance_vehicles_in_queues(setup_vehicles, mock_gv_db, mock_routing_map, mock_queues_manager, current_time):
    """Test vehicles that are in queues."""
    # Put first vehicle in queue
    mock_queues_manager.queues = {(0, 1): [setup_vehicles[0].id], (1, 2): set()}

    fcds, vehicles_moved = advance_vehicles_with_queues(
        [setup_vehicles[0]], current_time, mock_gv_db, mock_routing_map, mock_queues_manager, timedelta(seconds=0)
    )

    mock_queues_manager.batch_update.assert_called_once()


def test_advance_vehicles_batch_update_called(setup_vehicles, mock_gv_db, mock_routing_map, mock_queues_manager, current_time):
    """Test that batch_update is called at the end."""
    mock_queues_manager.queues = {(0, 1): set(), (1, 2): set()}

    advance_vehicles_with_queues(
        setup_vehicles, current_time, mock_gv_db, mock_routing_map, mock_queues_manager, timedelta(seconds=0)
    )

    mock_queues_manager.batch_update.assert_called_once()


def test_advance_vehicles_mixed_queue_status(setup_vehicles, mock_gv_db, mock_routing_map, mock_queues_manager, current_time):
    """Test with both vehicles in queues and not in queues."""
    vehicle1, vehicle2, vehicle3 = setup_vehicles
    # First vehicle in queue, others not
    mock_queues_manager.queues = {(0, 1): [vehicle1.id], (1, 2): set()}

    all_vehicles = [vehicle1, vehicle2, vehicle3]
    fcds, vehicles_moved = advance_vehicles_with_queues(
        all_vehicles, current_time, mock_gv_db, mock_routing_map, mock_queues_manager, timedelta(seconds=0)
    )

    # Some vehicles should have moved
    assert len(fcds) > 0
    mock_queues_manager.batch_update.assert_called_once()


def test_advance_vehicles_returns_tuple(setup_vehicles, mock_gv_db, mock_routing_map, mock_queues_manager, current_time):
    """Test that function returns correct tuple structure."""
    mock_queues_manager.queues = {(0, 1): set(), (1, 2): set()}

    result = advance_vehicles_with_queues(
        setup_vehicles, current_time, mock_gv_db, mock_routing_map, mock_queues_manager, timedelta(seconds=0)
    )

    assert isinstance(result, tuple)
    assert len(result) == 2
    fcds, vehicles_moved = result
    assert isinstance(fcds, list)
    assert isinstance(vehicles_moved, bool)


def test_advance_vehicles_with_los_tolerance(setup_vehicles, mock_gv_db, mock_routing_map, mock_queues_manager, current_time):
    """Test that los_vehicles_tolerance is passed through."""
    mock_queues_manager.queues = {(0, 1): set(), (1, 2): set()}
    custom_tolerance = timedelta(seconds=5)

    fcds, vehicles_moved = advance_vehicles_with_queues(
        setup_vehicles, current_time, mock_gv_db, mock_routing_map,
        mock_queues_manager, custom_tolerance
    )

    # Verify level_of_service was called (indicating parameters were passed)
    assert mock_gv_db.level_of_service_in_front_of_vehicle.called
