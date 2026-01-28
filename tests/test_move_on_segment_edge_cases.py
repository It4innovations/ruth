import pytest
from datetime import datetime, timedelta

from ruth.data.segment import LengthMeters, Segment, SpeedKph, SegmentPosition, speed_kph_to_mps, SpeedMps
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
def current_time():
    return datetime(2021, 1, 1, 0, 0, 0)


def test_move_on_segment_fractional_distance(setup_vehicle, setup_driving_route, current_time):
    """Test movement with fractional distance that doesn't reach segment end."""
    speed_mps = SpeedMps(speed_kph_to_mps(SpeedKph(10.0)))  # Low speed

    result_time, result_position, result_speed = move_on_segment(
        vehicle=setup_vehicle,
        driving_route_part=setup_driving_route,
        current_time=current_time,
        speed_mps=speed_mps,
        changed_segment=False
    )

    # Low speed should move only ~27.8 meters in 10 seconds
    expected_distance = speed_kph_to_mps(SpeedKph(10.0)) * 10
    assert result_position.position == LengthMeters(expected_distance)
    assert result_position.index == 0
    assert result_time == current_time + setup_vehicle.frequency


def test_move_on_segment_high_speed_reaches_end(setup_vehicle, setup_driving_route, current_time):
    """Test with high speed that reaches segment end."""
    # Set vehicle near end of segment
    setup_vehicle.set_position(SegmentPosition(index=0, position=LengthMeters(900.0)))
    speed_mps = SpeedMps(speed_kph_to_mps(SpeedKph(100.0)))

    result_time, result_position, result_speed = move_on_segment(
        vehicle=setup_vehicle,
        driving_route_part=setup_driving_route,
        current_time=current_time,
        speed_mps=speed_mps,
        changed_segment=False
    )

    # Should reach segment end at 1000
    expected_distance = speed_kph_to_mps(SpeedKph(100.0)) * 10 # 27.78 m/s * 10s = 277.8m, but capped at 1000m
    assert result_position.position == LengthMeters(1000.0)
    assert result_position.index == 0

    time_diff = result_time - current_time
    assert time_diff < setup_vehicle.frequency



def test_move_on_segment_segment_end_delta_behavior(setup_vehicle, setup_driving_route, current_time):
    """Test when vehicle movement ends exactly at segment boundary."""
    setup_vehicle.set_position(SegmentPosition(index=0, position=LengthMeters(860.0)))
    # Speed such that in 10s, vehicle moves exactly 140m to reach 1000m
    speed_mps = SpeedMps(speed_kph_to_mps(SpeedKph(50.0)))  # ~13.9 m/s, 139m in 10s

    result_time, result_position, result_speed = move_on_segment(
        vehicle=setup_vehicle,
        driving_route_part=setup_driving_route,
        current_time=current_time,
        speed_mps=speed_mps,
        changed_segment=False
    )

    # Should reach segment end
    # Starting at 860m + 139m = 999m, but rounded to 1000m
    assert result_position.position == LengthMeters(1000.0)


def test_move_on_segment_changed_segment_flag_true(setup_vehicle, setup_driving_route, current_time):
    """Test with changed_segment=True uses second segment."""
    setup_vehicle.set_position(SegmentPosition(index=0, position=LengthMeters(1000.0)))
    speed_mps = SpeedMps(speed_kph_to_mps(SpeedKph(50.0)))

    result_time, result_position, result_speed = move_on_segment(
        vehicle=setup_vehicle,
        driving_route_part=setup_driving_route,
        current_time=current_time,
        speed_mps=speed_mps,
        changed_segment=True
    )

    # Should be on second segment
    assert result_position.index == 1
    # Position should be measured from start of new segment
    assert result_position.position == LengthMeters(speed_kph_to_mps(SpeedKph(50.0)) * 10)  # Speed * Time (10 seconds)


def test_move_on_segment_preserves_speed(setup_vehicle, setup_driving_route, current_time):
    """Test that vehicle speed is preserved in return value."""
    speed_mps = SpeedMps(speed_kph_to_mps(SpeedKph(50.0)))

    result_time, result_position, result_speed = move_on_segment(
        vehicle=setup_vehicle,
        driving_route_part=setup_driving_route,
        current_time=current_time,
        speed_mps=speed_mps,
        changed_segment=False
    )

    assert result_speed == speed_mps