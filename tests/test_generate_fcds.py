import pytest
from unittest.mock import MagicMock, call
from datetime import datetime, timedelta

from ruth.data.map import Map
from ruth.data.segment import LengthMeters, Segment, SpeedKph, SegmentPosition, speed_kph_to_mps, SpeedMps
from ruth.simulator.route import generate_fcds
from ruth.simulator.simulation import FCDRecord
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
        status="test_status"
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


def test_generate_fcds_vehicle_moving(setup_vehicle, setup_driving_route, current_time):
    """Test FCD generation for vehicle moving on segment."""
    start_time = current_time
    end_time = current_time + timedelta(seconds=10)
    speed_mps = SpeedMps(speed_kph_to_mps(SpeedKph(50.0)))

    start_pos = SegmentPosition(index=0, position=LengthMeters(0.0))
    end_pos = SegmentPosition(index=0, position=LengthMeters(139.0))  # ~139m in 10s at 50 km/h

    fcds = generate_fcds(
        start_time, end_time, start_pos, end_pos, speed_mps,
        setup_vehicle, setup_driving_route, remains_active=True
    )

    assert len(fcds) >= 2  # At least start and end points
    assert fcds[0].datetime == start_time + setup_vehicle.fcd_sampling_period
    assert fcds[-1].datetime == end_time
    assert fcds[-1].active is True
    assert all(fcd.vehicle_id == 0 for fcd in fcds)
    assert all(fcd.status == "test_status" for fcd in fcds)


def test_generate_fcds_vehicle_stopped(setup_vehicle, setup_driving_route, current_time):
    """Test FCD generation for stopped vehicle."""
    start_time = current_time
    end_time = current_time + timedelta(seconds=10)
    speed_mps = SpeedMps(0.0)

    pos = SegmentPosition(index=0, position=LengthMeters(100.0))

    fcds = generate_fcds(
        start_time, end_time, pos, pos, speed_mps,
        setup_vehicle, setup_driving_route, remains_active=True
    )

    # For stopped vehicle with zero speed, records are still generated at sampling period
    assert len(fcds) >= 1
    # All records should be at the same position
    assert all(fcd.offset_from_start == LengthMeters(100.0) for fcd in fcds)
    assert all(fcd.vehicle_speed_mps == SpeedMps(0.0) for fcd in fcds)
    assert fcds[-1].datetime == end_time


def test_generate_fcds_vehicle_inactive(setup_vehicle, setup_driving_route, current_time):
    """Test FCD generation with vehicle becoming inactive."""
    start_time = current_time
    end_time = current_time + timedelta(seconds=5)
    speed_mps = SpeedMps(speed_kph_to_mps(SpeedKph(50.0)))

    start_pos = SegmentPosition(index=0, position=LengthMeters(0.0))
    end_pos = SegmentPosition(index=0, position=LengthMeters(70.0))

    fcds = generate_fcds(
        start_time, end_time, start_pos, end_pos, speed_mps,
        setup_vehicle, setup_driving_route, remains_active=False
    )

    assert fcds[-1].active is False


def test_generate_fcds_vehicle_changes_segment(setup_vehicle, setup_driving_route, current_time):
    """Test FCD generation when vehicle moves to next segment."""
    start_time = current_time
    end_time = current_time + timedelta(seconds=10)
    speed_mps = SpeedMps(speed_kph_to_mps(SpeedKph(50.0)))

    start_pos = SegmentPosition(index=0, position=LengthMeters(1000.0))  # At end of segment
    end_pos = SegmentPosition(index=1, position=LengthMeters(100.0))    # On next segment

    fcds = generate_fcds(
        start_time, end_time, start_pos, end_pos, speed_mps,
        setup_vehicle, setup_driving_route, remains_active=True
    )

    # Should have records on both segments
    assert fcds[-1].segment == setup_driving_route[1]


def test_generate_fcds_sampling_period(setup_vehicle, setup_driving_route, current_time):
    """Test that FCD records are generated at correct sampling period."""
    start_time = current_time
    end_time = current_time + timedelta(seconds=20)
    speed_mps = SpeedMps(speed_kph_to_mps(SpeedKph(50.0)))

    start_pos = SegmentPosition(index=0, position=LengthMeters(0.0))
    end_pos = SegmentPosition(index=0, position=LengthMeters(280.0))

    fcds = generate_fcds(
        start_time, end_time, start_pos, end_pos, speed_mps,
        setup_vehicle, setup_driving_route, remains_active=True
    )

    # Check spacing between records (should be fcd_sampling_period except last)
    for i in range(len(fcds) - 1):
        time_diff = fcds[i+1].datetime - fcds[i].datetime
        assert time_diff == setup_vehicle.fcd_sampling_period


def test_generate_fcds_empty_time_span(setup_vehicle, setup_driving_route, current_time):
    """Test FCD generation with no time span."""
    start_time = current_time
    end_time = current_time
    speed_mps = SpeedMps(speed_kph_to_mps(SpeedKph(50.0)))

    pos = SegmentPosition(index=0, position=LengthMeters(100.0))

    fcds = generate_fcds(
        start_time, end_time, pos, pos, speed_mps,
        setup_vehicle, setup_driving_route, remains_active=True
    )

    # Should have exactly one FCD at end time
    assert len(fcds) == 1
    assert fcds[0].datetime == end_time


def test_generate_fcds_correct_segment_references(setup_vehicle, setup_driving_route, current_time):
    """Test that FCD records reference correct segments."""
    start_time = current_time
    end_time = current_time + timedelta(seconds=10)
    speed_mps = SpeedMps(speed_kph_to_mps(SpeedKph(50.0)))

    start_pos = SegmentPosition(index=0, position=LengthMeters(0.0))
    end_pos = SegmentPosition(index=0, position=LengthMeters(139.0))

    fcds = generate_fcds(
        start_time, end_time, start_pos, end_pos, speed_mps,
        setup_vehicle, setup_driving_route, remains_active=True
    )

    # All FCD records should reference first segment
    assert all(fcd.segment == setup_driving_route[0] for fcd in fcds)


def test_fcd_conversion_profile_preserves_output_and_resets_step_metrics(
        tmp_path, setup_vehicle, setup_driving_route, current_time):
    import gc
    import h5py
    import numpy as np
    from ruth.fcd_history import FCDHistory
    from ruth.data.hdf_stream_writer import fcd_records_to_array

    records = generate_fcds(
        current_time, current_time + timedelta(seconds=10),
        SegmentPosition(0, LengthMeters(0)), SegmentPosition(0, LengthMeters(100)),
        SpeedMps(10), setup_vehicle, setup_driving_route, remains_active=True)
    callbacks = list(gc.callbacks)
    with FCDHistory(str(tmp_path / "profile.h5"), buffer_size=1,
                    max_records_per_file=1000) as history:
        history.extend(records)
        metrics = history.collect_step_metrics()
        assert metrics["fcd_records_flushed"] == len(records)
        assert metrics["fcd_convert_ms"] >= metrics["fcd_convert_gc_ms"] >= 0
        assert metrics["fcd_convert_cpu_ms"] >= 0
        assert all(metrics[f"fcd_convert_gc_gen{g}_collections"] >= 0 for g in range(3))
        assert history.collect_step_metrics() == {}
        assert history.collect_total_metrics() == metrics
    assert gc.callbacks == callbacks
    with h5py.File(tmp_path / "profile-part0000.h5", "r") as output:
        np.testing.assert_array_equal(output["fcd"][:], fcd_records_to_array(records))
