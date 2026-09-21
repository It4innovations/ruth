import pytest
from unittest.mock import MagicMock
from datetime import datetime, timedelta

from ruth.data.map import Map
from ruth.data.segment import LengthMeters, Segment, SpeedKph, SegmentPosition, speed_kph_to_mps, SpeedMps
from ruth.globalview import GlobalView
from ruth.simulator.route import get_vehicle_speed
from ruth.simulator.route import MovementInput
from ruth.simulator.route import VolumeToCapacityMovementModel, SegmentArrivals
from ruth.vehicle import Vehicle


@pytest.mark.parametrize("ahead,lanes,length,arrivals", [
    (0, 1, 1000, 0), (64, 1, 1000, 10), (64, 2, 1000, 10),
    (129, 1, 1000, 0), (150, 1, 1000, 0), (0, 1, 5, 0), (1, 1, 5, 0),
])
def test_improved_equation(ahead, lanes, length, arrivals, current_time):
    segment = Segment(0, 1, length, 50, lanes)
    gv = MagicMock(GlobalView)
    gv.number_of_vehicles_ahead.return_value = ahead
    routing_map = MagicMock(Map)
    routing_map.current_network = {0: {1: {"highway": "residential"}}}
    model = VolumeToCapacityMovementModel(gv, routing_map, timedelta(seconds=2))
    for vid in range(arrivals):
        model.arrivals.add(vid, segment.id, current_time - timedelta(seconds=1))
    factor = model.traffic_factor(current_time, segment, 999, 200, timedelta(seconds=2))
    expected = max(0, 1 - ((ahead + 1) / (length / 1000 * lanes * 130)) ** 2) if ahead else 1.0
    expected /= 1 + 0.15 * (arrivals * 60 / (500 * lanes)) ** 4
    assert factor == pytest.approx(expected)
    gv.number_of_vehicles_ahead.assert_called_once_with(
        current_time, segment.id, timedelta(seconds=2), 999, 200)


def test_segment_arrivals_window(current_time):
    state = SegmentArrivals()
    state.add(1, (0, 1), current_time - timedelta(seconds=61))
    state.add(1, (0, 1), current_time)  # Residence is not another arrival.
    state.add(2, (0, 1), current_time - timedelta(seconds=60))
    state.add(3, (0, 1), current_time - timedelta(seconds=59))
    state.add(3, (1, 0), current_time - timedelta(seconds=30))
    state.add(3, (0, 1), current_time - timedelta(seconds=20))  # Unique IDs.
    state.add(4, (0, 1), current_time + timedelta(seconds=1))
    state.drop_older_than(current_time)
    assert state.count_vehicles((0, 1), current_time) == 1
    assert state.count_vehicles((0, 1), current_time + timedelta(seconds=1)) == 2
    assert state.count_vehicles((1, 0), current_time) == 1


@pytest.mark.parametrize("with_arrivals", [False, True])
def test_arrival_checkpoint_compatibility(current_time, with_arrivals):
    import io
    import pickle
    from types import SimpleNamespace
    from ruth.simulator.simulation import Simulation
    from ruth.simulator.singlenode import Simulator

    # Avoid map downloads while exercising the actual checkpoint hooks.
    simulation = object.__new__(Simulation)
    simulation.setting = SimpleNamespace(round_freq=timedelta(seconds=5))
    simulation.vehicles = []
    simulation._routing_map = SimpleNamespace(
        current_network=SimpleNamespace(edges=lambda **kwargs: []))
    if with_arrivals:
        runtime = Simulator(simulation)
        runtime.segment_arrivals.add(7, (1, 2), current_time)
    payload = pickle.dumps(simulation)

    class LegacySimulation:
        def __setstate__(self, state):
            self.__dict__.update(state)

    class LegacyUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "ruth.simulator.simulation" and name == "Simulation":
                return LegacySimulation
            if module == "ruth.simulator.route":
                raise AssertionError("Old RUTH cannot load new movement classes")
            return super().find_class(module, name)

    legacy = LegacyUnpickler(io.BytesIO(payload)).load()
    if with_arrivals:
        assert legacy.segment_arrivals_state == {
            "last_segment": {7: (1, 2)},
            "arrivals": {(1, 2): [(current_time, 7)]},
        }
    restored = pickle.loads(payload)
    runtime = Simulator(restored)
    assert runtime.segment_arrivals.count_vehicles((1, 2), current_time) == int(with_arrivals)
    if with_arrivals:
        # Restoring last_segment prevents counting residence as a new entry.
        runtime.segment_arrivals.add(7, (1, 2), current_time + timedelta(seconds=1))
        assert len(restored.segment_arrivals_state["arrivals"][(1, 2)]) == 1


@pytest.mark.parametrize("closed", [False, True])
def test_improved_segment_transition(setup_vehicle, setup_driving_route, current_time, closed):
    setup_vehicle.set_position(SegmentPosition(0, 1000))
    setup_vehicle.has_next_segment_closed = lambda _: closed
    gv = MagicMock(GlobalView)
    gv.number_of_vehicles_ahead.return_value = 0
    routing_map = MagicMock(Map)
    routing_map.current_network = {1: {2: {"highway": "primary"}}}
    model = VolumeToCapacityMovementModel(gv, routing_map, timedelta(0))
    result, = model.compute_batch([MovementInput(setup_vehicle, current_time, setup_driving_route)])
    assert result.segment_pos.index == (0 if closed else 1)
    assert not model.arrivals.arrivals
    assert not model.arrivals.last_segment
    if closed:
        assert result.assigned_speed_mps == 0
        gv.number_of_vehicles_ahead.assert_not_called()
    else:
        assert result.assigned_speed_mps == pytest.approx(50 / 3.6)
        assert result.segment_pos.position == pytest.approx(result.assigned_speed_mps * 10)


@pytest.mark.parametrize("value,enabled", [("1", True), ("true", True), ("on", True),
                                           ("0", False), ("false", False), ("", False)])
def test_vtc_movement_flag(monkeypatch, value, enabled):
    from ruth.feature_flags import vtc_movement_enabled
    monkeypatch.setenv("RUTH_ENABLE_VTC_MOVEMENT", value)
    assert vtc_movement_enabled() is enabled


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
