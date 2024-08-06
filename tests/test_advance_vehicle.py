import pytest
from unittest.mock import MagicMock
from datetime import datetime, timedelta

from ruth.data.map import Map
from ruth.data.segment import LengthMeters, Segment, SpeedKph, speed_kph_to_mps, SpeedMps
from ruth.globalview import GlobalView
from ruth.simulator.route import advance_vehicle, advance_waiting_vehicle
from ruth.simulator.simulation import FCDRecord
from ruth.vehicle import Vehicle
from ruth.simulator.queues import QueuesManager


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
    segment1 = Segment(0, 1, LengthMeters(1000.0), SpeedKph(50.0))
    segment2 = Segment(1, 2, LengthMeters(1000.0), SpeedKph(50.0))
    return [segment1, segment2]


@pytest.fixture
def mock_gv_db():
    gv_db = MagicMock(GlobalView)
    gv_db.level_of_service_in_front_of_vehicle = MagicMock()
    return gv_db


@pytest.fixture
def mock_routing_map():
    routing_map = MagicMock(Map)
    routing_map.osm_route_to_py_segments = MagicMock(return_value=[Segment(0, 1, LengthMeters(1000.0), SpeedKph(50.0)),
                                                                   Segment(1, 2, LengthMeters(1000.0), SpeedKph(50.0))])
    return routing_map


@pytest.fixture
def mock_queues_manager():
    return MagicMock(QueuesManager)


@pytest.fixture
def current_time():
    return datetime(2021, 1, 1, 0, 0, 0)


def test_vehicle_advances_normally(setup_vehicle, mock_gv_db, mock_routing_map, mock_queues_manager, current_time):
    mock_gv_db.level_of_service_in_front_of_vehicle.return_value = 1.0
    mock_routing_map.get_current_max_speed = MagicMock(return_value=SpeedKph(50.0))

    fcd_records = advance_vehicle(
        vehicle=setup_vehicle,
        departure_time=current_time,
        gv_db=mock_gv_db,
        routing_map=mock_routing_map,
        queues_manager=mock_queues_manager
    )

    assert setup_vehicle.segment_position.index == 0
    assert (setup_vehicle.segment_position.position
            == LengthMeters(speed_kph_to_mps(SpeedKph(50.0)) * 10))  # Speed * Time (10 seconds)
    assert setup_vehicle.time_offset == timedelta(seconds=10)
    assert not mock_queues_manager.add_to_queue.called
    assert not mock_queues_manager.remove_vehicle.called
    assert fcd_records

    expected_fcd_records = [
        FCDRecord(
            datetime=current_time + timedelta(seconds=5),
            vehicle_id=0,
            segment=Segment(0, 1, LengthMeters(1000.0), SpeedKph(50.0)),  # Segment length 1000, current speed 50 km/h
            start_offset=LengthMeters(speed_kph_to_mps(SpeedKph(50.0)) * 5),  # Speed * Time (5 seconds
            speed=speed_kph_to_mps(SpeedKph(50.0)),
            status="",
            active=True
        ),
        FCDRecord(
            datetime=current_time + timedelta(seconds=10),
            vehicle_id=0,
            segment=Segment(0, 1, LengthMeters(1000.0), SpeedKph(50.0)),
            # Segment 0, length 1000, current speed 50 km/h
            start_offset=LengthMeters(speed_kph_to_mps(SpeedKph(50.0)) * 10),  # Speed * Time (10 seconds)
            speed=speed_kph_to_mps(SpeedKph(50.0)),
            status="",
            active=True
        )
    ]
    for i in range(2):
        assert fcd_records[i].datetime == expected_fcd_records[i].datetime
        assert fcd_records[i].vehicle_id == expected_fcd_records[i].vehicle_id
        assert fcd_records[i].segment == expected_fcd_records[i].segment
        assert fcd_records[i].start_offset == expected_fcd_records[i].start_offset
        assert fcd_records[i].speed == expected_fcd_records[i].speed
        assert fcd_records[i].status == expected_fcd_records[i].status
        assert fcd_records[i].active == expected_fcd_records[i].active


def test_vehicle_reaches_end_of_segment(setup_vehicle, mock_gv_db, mock_routing_map, mock_queues_manager, current_time):
    mock_gv_db.level_of_service_in_front_of_vehicle.return_value = 1.0
    mock_routing_map.get_current_max_speed = MagicMock(return_value=SpeedKph(50.0))
    setup_vehicle.start_distance_offset = LengthMeters(990.0)

    fcd_records = advance_vehicle(
        vehicle=setup_vehicle,
        departure_time=current_time,
        gv_db=mock_gv_db,
        routing_map=mock_routing_map,
        queues_manager=mock_queues_manager
    )

    assert setup_vehicle.segment_position.index == 0
    assert setup_vehicle.segment_position.position == LengthMeters(1000.0)
    assert mock_queues_manager.add_to_queue.called
    assert len(fcd_records) == 1

    time_travelled = LengthMeters(10.0) / speed_kph_to_mps(SpeedKph(50.0))
    assert fcd_records[0].datetime == current_time + timedelta(seconds=time_travelled)
    assert fcd_records[0].segment == Segment(0, 1, LengthMeters(1000.0), SpeedKph(50.0))
    assert fcd_records[0].start_offset == LengthMeters(1000.0)
    assert fcd_records[0].speed == speed_kph_to_mps(SpeedKph(50.0))


def test_vehicle_reaches_destination(setup_vehicle, mock_gv_db, mock_routing_map, mock_queues_manager, current_time):
    mock_gv_db.level_of_service_in_front_of_vehicle.return_value = 1.0
    mock_routing_map.get_current_max_speed = MagicMock(return_value=SpeedKph(50.0))
    mock_routing_map.osm_route_to_py_segments = MagicMock(return_value=[
        Segment(1, 2, LengthMeters(1000.0), SpeedKph(50.0))])

    setup_vehicle.start_distance_offset = LengthMeters(990.0)
    setup_vehicle.start_index = 1

    fcd_records = advance_vehicle(
        vehicle=setup_vehicle,
        departure_time=current_time,
        gv_db=mock_gv_db,
        routing_map=mock_routing_map,
        queues_manager=mock_queues_manager
    )

    assert setup_vehicle.segment_position.index == 1
    assert setup_vehicle.segment_position.position == LengthMeters(1000.0)
    assert not setup_vehicle.active
    assert mock_queues_manager.remove_inactive_vehicle.called
    assert not mock_queues_manager.add_to_queue.called
    assert len(fcd_records) == 1

    time_travelled = LengthMeters(10.0) / speed_kph_to_mps(SpeedKph(50.0))
    assert fcd_records[0].datetime == current_time + timedelta(seconds=time_travelled)
    assert fcd_records[0].segment == Segment(1, 2, LengthMeters(1000.0), SpeedKph(50.0))
    assert fcd_records[0].start_offset == LengthMeters(1000.0)
    assert fcd_records[0].speed == speed_kph_to_mps(SpeedKph(50.0))


def test_vehicle_advances_on_the_next_segment(setup_vehicle, mock_gv_db, mock_routing_map, mock_queues_manager,
                                              current_time):
    mock_gv_db.level_of_service_in_front_of_vehicle.return_value = 1.0
    mock_routing_map.get_current_max_speed = MagicMock(return_value=SpeedKph(50.0))
    setup_vehicle.start_distance_offset = LengthMeters(1000.0)

    fcd_records = advance_vehicle(
        vehicle=setup_vehicle,
        departure_time=current_time,
        gv_db=mock_gv_db,
        routing_map=mock_routing_map,
        queues_manager=mock_queues_manager
    )

    assert setup_vehicle.segment_position.index == 1
    assert (setup_vehicle.segment_position.position
            == LengthMeters(speed_kph_to_mps(SpeedKph(50.0)) * 10))  # Speed * Time (10 seconds)
    assert setup_vehicle.time_offset == timedelta(seconds=10)
    assert not mock_queues_manager.add_to_queue.called
    assert mock_queues_manager.remove_vehicle.called
    assert fcd_records

    expected_fcd_records = [
        FCDRecord(
            datetime=current_time + timedelta(seconds=5),
            vehicle_id=0,
            segment=Segment(1, 2, LengthMeters(1000.0), SpeedKph(50.0)),
            start_offset=LengthMeters(speed_kph_to_mps(SpeedKph(50.0)) * 5),  # Speed * Time (5 seconds)
            speed=speed_kph_to_mps(SpeedKph(50.0)),
            status="",
            active=True
        ),
        FCDRecord(
            datetime=current_time + timedelta(seconds=10),
            vehicle_id=0,
            segment=Segment(1, 2, LengthMeters(1000.0), SpeedKph(50.0)),
            # Segment 0, length 1000, current speed 50 km/h
            start_offset=LengthMeters(speed_kph_to_mps(SpeedKph(50.0)) * 10),  # Speed * Time (10 seconds)
            speed=speed_kph_to_mps(SpeedKph(50.0)),
            status="",
            active=True
        )
    ]
    for i in range(2):
        assert fcd_records[i].datetime == expected_fcd_records[i].datetime
        assert fcd_records[i].vehicle_id == expected_fcd_records[i].vehicle_id
        assert fcd_records[i].segment == expected_fcd_records[i].segment
        assert fcd_records[i].start_offset == expected_fcd_records[i].start_offset
        assert fcd_records[i].speed == expected_fcd_records[i].speed
        assert fcd_records[i].status == expected_fcd_records[i].status
        assert fcd_records[i].active == expected_fcd_records[i].active


def test_vehicle_stuck_in_traffic(setup_vehicle, mock_gv_db, mock_routing_map, mock_queues_manager, current_time):
    mock_gv_db.level_of_service_in_front_of_vehicle.return_value = 0.0

    expected_position = setup_vehicle.segment_position
    expected_time_offset = setup_vehicle.time_offset + timedelta(seconds=10)

    fcd_records = advance_vehicle(
        vehicle=setup_vehicle,
        departure_time=current_time,
        gv_db=mock_gv_db,
        routing_map=mock_routing_map,
        queues_manager=mock_queues_manager
    )

    assert setup_vehicle.segment_position == expected_position
    assert setup_vehicle.time_offset == expected_time_offset

    expected_fcd_records = [
        FCDRecord(
            datetime=current_time + timedelta(seconds=5),
            vehicle_id=0,
            segment=Segment(0, 1, LengthMeters(1000.0), SpeedKph(50.0)),
            start_offset=expected_position.position,
            speed=SpeedMps(0.0),
            status="",
            active=True
        ),
        FCDRecord(
            datetime=current_time + timedelta(seconds=10),
            vehicle_id=0,
            segment=Segment(0, 1, LengthMeters(1000.0), SpeedKph(50.0)),
            start_offset=expected_position.position,
            speed=SpeedMps(0.0),
            status="",
            active=True
        )
    ]

    for i in range(2):
        assert fcd_records[i].datetime == expected_fcd_records[i].datetime
        assert fcd_records[i].vehicle_id == expected_fcd_records[i].vehicle_id
        assert fcd_records[i].segment == expected_fcd_records[i].segment
        assert fcd_records[i].start_offset == expected_fcd_records[i].start_offset
        assert fcd_records[i].speed == expected_fcd_records[i].speed
        assert fcd_records[i].status == expected_fcd_records[i].status
        assert fcd_records[i].active == expected_fcd_records[i].active


def test_advance_vehicle_with_los_change(setup_vehicle, mock_gv_db, mock_routing_map, mock_queues_manager,
                                         current_time):
    mock_gv_db.level_of_service_in_front_of_vehicle.return_value = 0.5
    custom_tolerance = timedelta(seconds=5)

    fcd_records = advance_vehicle(
        vehicle=setup_vehicle,
        departure_time=current_time,
        gv_db=mock_gv_db,
        routing_map=mock_routing_map,
        queues_manager=mock_queues_manager,
        los_vehicles_tolerance=custom_tolerance
    )

    expected_position = LengthMeters(speed_kph_to_mps(SpeedKph(25.0)) * 10)  # Adjusted speed * Time (10 seconds)
    assert setup_vehicle.segment_position.position == expected_position
    assert fcd_records  # Check if FCD records are generated


def test_advance_waiting_vehicle(setup_vehicle, mock_gv_db, mock_routing_map, mock_queues_manager, current_time):
    mock_gv_db.level_of_service_in_front_of_vehicle.return_value = 1.0

    expected_position = setup_vehicle.segment_position
    expected_time_offset = timedelta(seconds=10)

    fcd_records = advance_waiting_vehicle(
        vehicle=setup_vehicle,
        departure_time=current_time,
        routing_map=mock_routing_map,
    )

    assert fcd_records
    assert not mock_queues_manager.add_to_queue.called
    assert not mock_queues_manager.remove_vehicle.called

    assert setup_vehicle.segment_position == expected_position
    assert setup_vehicle.time_offset == expected_time_offset

    expected_fcd_records = [
        FCDRecord(
            datetime=current_time + timedelta(seconds=5),
            vehicle_id=0,
            segment=Segment(0, 1, LengthMeters(1000.0), SpeedKph(50.0)),
            start_offset=expected_position.position,
            speed=SpeedMps(0.0),
            status="",
            active=True
        ),
        FCDRecord(
            datetime=current_time + timedelta(seconds=10),
            vehicle_id=0,
            segment=Segment(0, 1, LengthMeters(1000.0), SpeedKph(50.0)),
            start_offset=expected_position.position,
            speed=SpeedMps(0.0),
            status="",
            active=True
        )
    ]

    for i in range(2):
        assert fcd_records[i].datetime == expected_fcd_records[i].datetime
        assert fcd_records[i].vehicle_id == expected_fcd_records[i].vehicle_id
        assert fcd_records[i].segment == expected_fcd_records[i].segment
        assert fcd_records[i].start_offset == expected_fcd_records[i].start_offset
        assert fcd_records[i].speed == expected_fcd_records[i].speed
        assert fcd_records[i].status == expected_fcd_records[i].status
        assert fcd_records[i].active == expected_fcd_records[i].active
