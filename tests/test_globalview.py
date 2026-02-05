"""Tests for GlobalView class to ensure proper unit conversions and speed calculations."""
import pytest
from datetime import datetime, timedelta

from ruth.data.segment import LengthMeters, Segment, SpeedKph, SpeedMps, SegmentId, speed_kph_to_mps
from ruth.globalview import GlobalView
from ruth.simulator.simulation import FCDRecord


@pytest.fixture
def current_time():
    return datetime(2024, 1, 1, 12, 0, 0)


@pytest.fixture
def sample_segment():
    return Segment(
        node_from=100,
        node_to=200,
        length=LengthMeters(1000.0),
        max_allowed_speed_kph=SpeedKph(50.0),
        lanes=2
    )


@pytest.fixture
def global_view():
    return GlobalView()


def test_get_segment_speed_single_vehicle(global_view, sample_segment, current_time):
    speed_kph = SpeedKph(50.0)
    speed_mps = SpeedMps(speed_kph_to_mps(speed_kph))

    fcd = FCDRecord(
        datetime=current_time,
        vehicle_id=1,
        segment=sample_segment,
        offset_from_start=LengthMeters(500.0),
        vehicle_speed_mps=speed_mps,
        status="",
        active=True
    )

    global_view.add(fcd)
    result_speed = global_view.get_segment_speed(sample_segment.id)

    # The result should be in km/h and should match the original speed
    assert result_speed is not None
    assert abs(result_speed - speed_kph) < 0.01, f"Expected {speed_kph} km/h, got {result_speed} km/h"


def test_get_segment_speed_multiple_vehicles(global_view, sample_segment, current_time):
    """Test averaging speeds from multiple vehicles with correct unit conversion."""
    # Vehicle 1: 50 km/h = 13.89 m/s
    # Vehicle 2: 60 km/h = 16.67 m/s
    # Average should be 55 km/h

    speeds_kph = [SpeedKph(50.0), SpeedKph(60.0)]
    expected_avg_kph = SpeedKph(55.0)

    for i, speed_kph in enumerate(speeds_kph):
        speed_mps = SpeedMps(speed_kph_to_mps(speed_kph))
        fcd = FCDRecord(
            datetime=current_time,
            vehicle_id=i,
            segment=sample_segment,
            offset_from_start=LengthMeters(500.0 + i * 10),
            vehicle_speed_mps=speed_mps,
            status="",
            active=True
        )
        global_view.add(fcd)

    result_speed = global_view.get_segment_speed(sample_segment.id)

    assert result_speed is not None
    assert abs(result_speed - expected_avg_kph) < 0.01, f"Expected {expected_avg_kph} km/h, got {result_speed} km/h"


def test_get_segment_speed_empty_segment(global_view):
    """Test that empty segments return None."""
    empty_segment_id = SegmentId((999, 888))
    result = global_view.get_segment_speed(empty_segment_id)
    assert result is None


def test_take_segment_speeds_returns_modified_segments(global_view, sample_segment, current_time):
    """Test that take_segment_speeds only returns modified segments with correct speeds."""
    speed_kph = SpeedKph(72.0)  # 72 km/h = 20 m/s
    speed_mps = SpeedMps(speed_kph_to_mps(speed_kph))

    fcd = FCDRecord(
        datetime=current_time,
        vehicle_id=1,
        segment=sample_segment,
        offset_from_start=LengthMeters(500.0),
        vehicle_speed_mps=speed_mps,
        status="",
        active=True
    )

    global_view.add(fcd)
    speeds = global_view.take_segment_speeds()

    assert sample_segment.id in speeds
    assert abs(speeds[sample_segment.id] - speed_kph) < 0.01

    # After taking speeds, modified_segments should be cleared
    speeds_again = global_view.take_segment_speeds()
    assert len(speeds_again) == 0

    # Explicitly verify the modified_segments set is empty
    assert len(global_view.modified_segments) == 0


def test_vehicle_speed_updates_correctly(global_view, sample_segment, current_time):
    """Test that when a vehicle changes speed, the latest speed is used."""
    # Vehicle starts at 50 km/h
    initial_speed_kph = SpeedKph(50.0)
    fcd1 = FCDRecord(
        datetime=current_time,
        vehicle_id=1,
        segment=sample_segment,
        offset_from_start=LengthMeters(100.0),
        vehicle_speed_mps=SpeedMps(speed_kph_to_mps(initial_speed_kph)),
        status="",
        active=True
    )
    global_view.add(fcd1)

    # Same vehicle later at 30 km/h (slowed down)
    updated_speed_kph = SpeedKph(30.0)
    fcd2 = FCDRecord(
        datetime=current_time + timedelta(seconds=10),
        vehicle_id=1,  # Same vehicle
        segment=sample_segment,
        offset_from_start=LengthMeters(200.0),
        vehicle_speed_mps=SpeedMps(speed_kph_to_mps(updated_speed_kph)),
        status="",
        active=True
    )
    global_view.add(fcd2)

    result = global_view.get_segment_speed(sample_segment.id)

    assert result is not None
    assert abs(result - updated_speed_kph) < 0.01, f"Expected {updated_speed_kph} km/h, got {result} km/h"


def test_vehicle_changes_segments(global_view, current_time):
    segment1 = Segment(node_from=100, node_to=200,
                       length=LengthMeters(1000.0), max_allowed_speed_kph=SpeedKph(50.0), lanes=2)
    segment2 = Segment(node_from=200, node_to=300,
                          length=LengthMeters(800.0), max_allowed_speed_kph=SpeedKph(60.0), lanes=2)

    # Vehicle starts on segment1
    vehicle_id=10
    speed_kph = SpeedKph(50.0)
    fcd1 = FCDRecord(
        datetime=current_time,
        vehicle_id=vehicle_id,
        segment=segment1,
        offset_from_start=LengthMeters(500.0),
        vehicle_speed_mps=SpeedMps(speed_kph_to_mps(speed_kph)),
        status="",
        active=True
    )
    global_view.add(fcd1)

    assert segment1.id in global_view.modified_segments
    assert segment1.id in global_view.car_to_segment.values()

    # Vehicle moves to segment2
    fcd2 = FCDRecord(
        datetime=current_time + timedelta(seconds=10),
        vehicle_id=vehicle_id,
        segment=segment2,
        offset_from_start=LengthMeters(100.0),
        vehicle_speed_mps=SpeedMps(speed_kph_to_mps(speed_kph)),
        status="",
        active=True
    )
    global_view.add(fcd2)

    assert segment1.id in global_view.modified_segments, "Old segment should be marked as modified"
    assert segment2.id in global_view.modified_segments, "New segment should be marked as modified"
    assert global_view.car_to_segment[vehicle_id] == segment2.id

    # Segment1 should have no vehicles, segment2 should have the vehicle
    assert len(global_view.fcd_by_segment[segment1.id]) == 1
    assert len(global_view.fcd_by_segment[segment2.id]) == 1


def test_drop_old_removes_old_fcds(global_view, sample_segment, current_time):
    times = [
        current_time,
        current_time + timedelta(seconds=30),
        current_time + timedelta(seconds=60)
    ]

    for i, time in enumerate(times):
        fcd = FCDRecord(
            datetime=time,
            vehicle_id=i,
            segment=sample_segment,
            offset_from_start=LengthMeters(500.0),
            vehicle_speed_mps=SpeedMps(speed_kph_to_mps(SpeedKph(50.0))),
            status="",
            active=True
        )
        global_view.add(fcd)

    # Verify all three records are present
    assert len(global_view.fcd_by_segment[sample_segment.id]) == 3

    # Drop records older than current_time + 45 seconds
    threshold = current_time + timedelta(seconds=45)
    global_view.drop_old(threshold)

    # Only the last record should remain (at 60 seconds)
    assert len(global_view.fcd_by_segment[sample_segment.id]) == 1
    remaining_fcd = global_view.fcd_by_segment[sample_segment.id][0]
    assert remaining_fcd.datetime == times[2]


def test_drop_old_marks_segments_as_modified(global_view, sample_segment, current_time):
    """Test that drop_old marks segments as modified when FCDs are removed."""
    # Add FCD records
    fcd1 = FCDRecord(
        datetime=current_time,
        vehicle_id=1,
        segment=sample_segment,
        offset_from_start=LengthMeters(500.0),
        vehicle_speed_mps=SpeedMps(speed_kph_to_mps(SpeedKph(50.0))),
        status="",
        active=True
    )
    global_view.add(fcd1)

    fcd2 = FCDRecord(
        datetime=current_time + timedelta(seconds=60),
        vehicle_id=2,
        segment=sample_segment,
        offset_from_start=LengthMeters(600.0),
        vehicle_speed_mps=SpeedMps(speed_kph_to_mps(SpeedKph(50.0))),
        status="",
        active=True
    )
    global_view.add(fcd2)

    # Clear modified segments
    global_view.modified_segments.clear()

    # Drop old records
    threshold = current_time + timedelta(seconds=45)
    global_view.drop_old(threshold)

    # Segment should be marked as modified because FCDs were removed
    assert sample_segment.id in global_view.modified_segments


def test_drop_old_does_not_mark_unchanged_segments(global_view, sample_segment, current_time):
    """Test that drop_old does not mark segments as modified if no FCDs are removed."""
    # Add FCD record
    fcd = FCDRecord(
        datetime=current_time + timedelta(seconds=60),
        vehicle_id=1,
        segment=sample_segment,
        offset_from_start=LengthMeters(500.0),
        vehicle_speed_mps=SpeedMps(speed_kph_to_mps(SpeedKph(50.0))),
        status="",
        active=True
    )
    global_view.add(fcd)

    # Clear modified segments
    global_view.modified_segments.clear()

    # Drop old records with a threshold that doesn't remove anything
    threshold = current_time + timedelta(seconds=45)
    global_view.drop_old(threshold)

    # Segment should NOT be marked as modified because no FCDs were removed
    assert sample_segment.id not in global_view.modified_segments



def test_take_segment_speeds_multiple_segments(global_view, current_time):
    """Test that take_segment_speeds returns all modified segments."""
    # Create three segments
    segment1 = Segment(
        node_from=100,
        node_to=200,
        length=LengthMeters(1000.0),
        max_allowed_speed_kph=SpeedKph(50.0),
        lanes=2
    )
    segment2 = Segment(
        node_from=200,
        node_to=300,
        length=LengthMeters(1000.0),
        max_allowed_speed_kph=SpeedKph(50.0),
        lanes=2
    )
    segment3 = Segment(
        node_from=300,
        node_to=400,
        length=LengthMeters(1000.0),
        max_allowed_speed_kph=SpeedKph(50.0),
        lanes=2
    )

    # Add vehicles to all three segments
    speeds_kph = [SpeedKph(30.0), SpeedKph(40.0), SpeedKph(50.0)]
    segments = [segment1, segment2, segment3]

    for i, (segment, speed) in enumerate(zip(segments, speeds_kph)):
        fcd = FCDRecord(
            datetime=current_time,
            vehicle_id=i,
            segment=segment,
            offset_from_start=LengthMeters(500.0),
            vehicle_speed_mps=SpeedMps(speed_kph_to_mps(speed)),
            status="",
            active=True
        )
        global_view.add(fcd)

    # Get all modified segments
    speeds = global_view.take_segment_speeds()

    # All three segments should be in the result
    assert len(speeds) == 3
    assert segment1.id in speeds
    assert segment2.id in speeds
    assert segment3.id in speeds

    # Check speeds are correct
    assert abs(speeds[segment1.id] - speeds_kph[0]) < 0.01
    assert abs(speeds[segment2.id] - speeds_kph[1]) < 0.01
    assert abs(speeds[segment3.id] - speeds_kph[2]) < 0.01

    # After taking speeds, modified_segments should be cleared
    speeds_again = global_view.take_segment_speeds()
    assert len(speeds_again) == 0


def test_take_segment_speeds_includes_empty_segments(global_view, current_time):
    """Test that take_segment_speeds includes segments that became empty (None speed)."""
    segment = Segment(
        node_from=100,
        node_to=200,
        length=LengthMeters(1000.0),
        max_allowed_speed_kph=SpeedKph(50.0),
        lanes=2
    )

    # Add a vehicle
    fcd = FCDRecord(
        datetime=current_time,
        vehicle_id=1,
        segment=segment,
        offset_from_start=LengthMeters(500.0),
        vehicle_speed_mps=SpeedMps(speed_kph_to_mps(SpeedKph(50.0))),
        status="",
        active=True
    )
    global_view.add(fcd)

    # Take speeds (clears modified_segments)
    speeds = global_view.take_segment_speeds()
    assert segment.id in speeds
    assert speeds[segment.id] is not None

    # Drop old FCDs to make the segment empty
    threshold = current_time + timedelta(seconds=60)
    global_view.drop_old(threshold)

    # The segment should be marked as modified
    speeds_after_drop = global_view.take_segment_speeds()

    # The segment should be in the result with None speed (no vehicles)
    assert segment.id in speeds_after_drop
    assert speeds_after_drop[segment.id] is None



def test_modified_segments_cleared_after_take_segment_speeds(global_view, current_time):
    """Test that modified_segments is properly cleared after take_segment_speeds() in various scenarios."""
    segment1 = Segment(
        node_from=100,
        node_to=200,
        length=LengthMeters(1000.0),
        max_allowed_speed_kph=SpeedKph(50.0),
        lanes=2
    )
    segment2 = Segment(
        node_from=200,
        node_to=300,
        length=LengthMeters(1000.0),
        max_allowed_speed_kph=SpeedKph(50.0),
        lanes=2
    )

    # Initially, modified_segments should be empty
    assert len(global_view.modified_segments) == 0

    # Add vehicle to segment1
    fcd1 = FCDRecord(
        datetime=current_time,
        vehicle_id=1,
        segment=segment1,
        offset_from_start=LengthMeters(500.0),
        vehicle_speed_mps=SpeedMps(speed_kph_to_mps(SpeedKph(50.0))),
        status="",
        active=True
    )
    global_view.add(fcd1)

    # modified_segments should contain segment1
    assert len(global_view.modified_segments) == 1
    assert segment1.id in global_view.modified_segments

    # Add vehicle to segment2
    fcd2 = FCDRecord(
        datetime=current_time,
        vehicle_id=2,
        segment=segment2,
        offset_from_start=LengthMeters(500.0),
        vehicle_speed_mps=SpeedMps(speed_kph_to_mps(SpeedKph(40.0))),
        status="",
        active=True
    )
    global_view.add(fcd2)

    # modified_segments should contain both segments
    assert len(global_view.modified_segments) == 2
    assert segment1.id in global_view.modified_segments
    assert segment2.id in global_view.modified_segments

    # Call take_segment_speeds
    speeds = global_view.take_segment_speeds()

    # Both segments should be in the returned speeds
    assert len(speeds) == 2

    # modified_segments should be completely cleared
    assert len(global_view.modified_segments) == 0
    assert segment1.id not in global_view.modified_segments
    assert segment2.id not in global_view.modified_segments

    # Calling take_segment_speeds again should return empty dict
    speeds_again = global_view.take_segment_speeds()
    assert len(speeds_again) == 0
    assert len(global_view.modified_segments) == 0

    # Add new FCD to segment1
    fcd3 = FCDRecord(
        datetime=current_time + timedelta(seconds=10),
        vehicle_id=1,
        segment=segment1,
        offset_from_start=LengthMeters(600.0),
        vehicle_speed_mps=SpeedMps(speed_kph_to_mps(SpeedKph(30.0))),
        status="",
        active=True
    )
    global_view.add(fcd3)

    # Only segment1 should be in modified_segments
    assert len(global_view.modified_segments) == 1
    assert segment1.id in global_view.modified_segments
    assert segment2.id not in global_view.modified_segments

    # Take speeds again
    speeds_final = global_view.take_segment_speeds()
    assert len(speeds_final) == 1
    assert segment1.id in speeds_final

    # modified_segments should be cleared again
    assert len(global_view.modified_segments) == 0


def test_modified_segments_cleared_after_drop_old_and_take_speeds(global_view, sample_segment, current_time):
    """Test that modified_segments is cleared properly after drop_old followed by take_segment_speeds."""
    # Add old FCD
    fcd_old = FCDRecord(
        datetime=current_time,
        vehicle_id=1,
        segment=sample_segment,
        offset_from_start=LengthMeters(500.0),
        vehicle_speed_mps=SpeedMps(speed_kph_to_mps(SpeedKph(50.0))),
        status="",
        active=True
    )
    global_view.add(fcd_old)

    # Add recent FCD
    fcd_recent = FCDRecord(
        datetime=current_time + timedelta(seconds=60),
        vehicle_id=2,
        segment=sample_segment,
        offset_from_start=LengthMeters(600.0),
        vehicle_speed_mps=SpeedMps(speed_kph_to_mps(SpeedKph(40.0))),
        status="",
        active=True
    )
    global_view.add(fcd_recent)

    # Clear modified_segments
    global_view.take_segment_speeds()
    assert len(global_view.modified_segments) == 0

    # Drop old FCDs
    threshold = current_time + timedelta(seconds=45)
    global_view.drop_old(threshold)

    # Segment should be in modified_segments after drop_old
    assert len(global_view.modified_segments) == 1
    assert sample_segment.id in global_view.modified_segments

    # Take speeds
    speeds = global_view.take_segment_speeds()
    assert sample_segment.id in speeds

    # modified_segments should be cleared
    assert len(global_view.modified_segments) == 0


