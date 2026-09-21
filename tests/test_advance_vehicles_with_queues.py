import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, call
from datetime import datetime, timedelta

from ruth.data.map import Map
from ruth.data.segment import LengthMeters, Segment, SpeedKph, SegmentPosition, speed_kph_to_mps, SpeedMps
from ruth.globalview import GlobalView
from ruth.globalview_heterogeneous import GlobalView as HeterogeneousGlobalView
from ruth.simulator.route import MovementResult, advance_vehicles_with_queues
from ruth.simulator.route_heterogeneous import HeterogeneousMovementModel
from ruth.simulator.queues import QueuesManager
from ruth.vehicle import Vehicle
from ruth.vehicle_types import DEFAULT_VEHICLE_CLASSES


@pytest.mark.parametrize("improved,heterogeneous", [(False, False), (True, False),
                                                   (False, True), (True, True)])
def test_simulator_movement_switches(monkeypatch, current_time, improved, heterogeneous):
    from ruth.simulator import singlenode
    from ruth.simulator.route import VolumeToCapacityMovementModel

    monkeypatch.setenv("RUTH_ENABLE_VTC_MOVEMENT", "1" if improved else "0")
    monkeypatch.setattr(singlenode, "HeterogeneousMovementModel",
                        HeterogeneousMovementModel if heterogeneous else None)
    captured = {}

    def advance(*args, movement_model):
        captured["model"] = movement_model
        return [], False

    monkeypatch.setattr(singlenode, "advance_vehicles_with_queues", advance)
    sim = SimpleNamespace(
        global_view=GlobalView(), routing_map=MagicMock(Map), queues_manager=QueuesManager(),
        setting=SimpleNamespace(departure_time=current_time, los_vehicles_tolerance=timedelta(0)),
        compute_current_offset=lambda: timedelta(0))
    if improved and heterogeneous:
        with pytest.raises(ValueError, match="mutually exclusive"):
            singlenode.Simulator(sim)
        assert not captured
        return
    simulator = singlenode.Simulator(sim)
    assert simulator.advance_vehicles([]) == ([], False)
    assert not hasattr(sim, "segment_arrivals")
    model = captured["model"]
    if improved:
        assert isinstance(model, VolumeToCapacityMovementModel)
        simulator.advance_vehicles([])
        assert captured["model"].arrivals is model.arrivals
    elif heterogeneous:
        assert isinstance(model, HeterogeneousMovementModel)
    else:
        assert model is None


@pytest.mark.parametrize("transition", [False, True])
def test_simulator_publishes_arrivals_after_movement(monkeypatch, current_time, transition):
    from ruth.simulator import singlenode

    monkeypatch.setenv("RUTH_ENABLE_VTC_MOVEMENT", "1")
    vehicle = SimpleNamespace(id=1, current_node=0, next_node=1, time_offset=timedelta(0),
                              segment_position=SegmentPosition(0, 1000 if transition else 0))
    sim = SimpleNamespace(
        global_view=GlobalView(), routing_map=MagicMock(Map), queues_manager=QueuesManager(),
        setting=SimpleNamespace(departure_time=current_time, los_vehicles_tolerance=timedelta(0)),
        compute_current_offset=lambda: timedelta(0))
    simulator = singlenode.Simulator(sim)

    def advance(*args, movement_model):
        assert not movement_model.arrivals.arrivals
        if transition:
            vehicle.current_node, vehicle.next_node = 1, 2
        vehicle.segment_position = SegmentPosition(int(transition), 100)
        vehicle.time_offset += timedelta(seconds=10)
        return [], True

    monkeypatch.setattr(singlenode, "advance_vehicles_with_queues", advance)
    simulator.advance_vehicles([vehicle])
    assert not simulator.segment_arrivals.arrivals
    assert not sim.segment_arrivals_state["arrivals"]
    simulator.update_segment_arrivals()
    assert sim.segment_arrivals_state["arrivals"][(0, 1)] == [(current_time, 1)]
    assert simulator.segment_arrivals.count_vehicles((0, 1), current_time) == 1
    assert simulator.segment_arrivals.count_vehicles((1, 2), current_time) == int(transition)
    assert not hasattr(sim, "segment_arrivals")
    simulator.update_segment_arrivals()
    assert len(simulator.segment_arrivals.arrivals[(0, 1)]) == 1


def test_simulator_counts_entry_after_blocked_departure(
        monkeypatch, current_time, setup_vehicles, mock_routing_map):
    from ruth.simulator import singlenode

    monkeypatch.setenv("RUTH_ENABLE_VTC_MOVEMENT", "1")
    monkeypatch.setattr(singlenode, "HeterogeneousMovementModel", None)
    vehicle = setup_vehicles[0]
    gv = MagicMock(GlobalView)
    gv.number_of_vehicles_ahead.return_value = 130  # Saturate the 1 km segment.
    mock_routing_map.current_network = {0: {1: {"highway": "residential"}}}
    sim = SimpleNamespace(
        global_view=gv, routing_map=mock_routing_map, queues_manager=QueuesManager(),
        setting=SimpleNamespace(departure_time=current_time, los_vehicles_tolerance=timedelta(0)),
        compute_current_offset=lambda: timedelta(0))
    simulator = singlenode.Simulator(sim)

    # Wait longer than the arrival window, using the actual movement path.
    for _ in range(7):
        assert simulator.advance_vehicles([vehicle]) == ([], False)
        simulator.current_offset = vehicle.time_offset
        simulator.update_segment_arrivals()
        assert not simulator.segment_arrivals.arrivals
        assert not simulator.segment_arrivals.last_segment

    entry_time = current_time + vehicle.time_offset
    gv.number_of_vehicles_ahead.return_value = 0
    fcds, moved = simulator.advance_vehicles([vehicle])
    assert moved and fcds
    assert vehicle.start_distance_offset > 0
    assert not simulator.segment_arrivals.arrivals  # Publish after the batch.
    simulator.update_segment_arrivals()
    assert simulator.segment_arrivals.arrivals[(0, 1)] == [(entry_time, vehicle.id)]
    assert simulator.segment_arrivals.count_vehicles((0, 1), entry_time) == 1

    # Residence and later movement on the same segment are not new entries.
    gv.number_of_vehicles_ahead.return_value = 130
    simulator.advance_vehicles([vehicle])
    simulator.update_segment_arrivals()
    gv.number_of_vehicles_ahead.return_value = 0
    simulator.advance_vehicles([vehicle])
    simulator.update_segment_arrivals()
    assert simulator.segment_arrivals.arrivals[(0, 1)] == [(entry_time, vehicle.id)]


@pytest.mark.parametrize("length", [1000, 5, 1000 / 130, 0.1])
def test_vtc_model_lone_vehicle_moves(setup_vehicles, current_time, length):
    from ruth.simulator.route import VolumeToCapacityMovementModel, MovementInput
    vehicle = setup_vehicles[0]
    vehicle.vehicle_type = "truck"
    segment = Segment(0, 1, length, 50, 1)
    gv = GlobalView()
    routing_map = MagicMock(Map)
    routing_map.current_network = {0: {1: {"highway": "residential"}}}
    model = VolumeToCapacityMovementModel(gv, routing_map, timedelta(0))
    result, = model.compute_batch([MovementInput(vehicle, current_time, [segment])])
    assert result.assigned_speed_mps > 0
    assert result.segment_pos.position > 0
    factor = 1.0  # This step's arrivals are not yet visible.
    assert result.assigned_speed_mps == pytest.approx(50 / 3.6 * factor)


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


def test_advance_vehicles_batches_non_queued_movement_inputs(
        setup_vehicles, mock_gv_db, mock_routing_map, mock_queues_manager, current_time):
    class BatchRecordingMovementModel:
        def __init__(self):
            self.batch_sizes = []

        def compute_batch(self, movement_inputs):
            self.batch_sizes.append(len(movement_inputs))
            return [
                MovementResult(
                    movement_input.current_time + movement_input.vehicle.frequency,
                    SegmentPosition(movement_input.vehicle.segment_position.index, LengthMeters(10.0)),
                    SpeedMps(1.0),
                )
                for movement_input in movement_inputs
            ]

    movement_model = BatchRecordingMovementModel()

    advance_vehicles_with_queues(
        setup_vehicles,
        current_time,
        mock_gv_db,
        mock_routing_map,
        mock_queues_manager,
        timedelta(seconds=0),
        movement_model=movement_model,
    )

    assert movement_model.batch_sizes == [len(setup_vehicles)]


def test_advance_vehicles_uses_heterogeneous_movement_with_shared_queue_logic(current_time):
    vehicle = Vehicle(
        id=1,
        time_offset=timedelta(seconds=0),
        frequency=timedelta(seconds=10),
        start_index=0,
        start_distance_offset=LengthMeters(0.0),
        origin_node=0,
        dest_node=2,
        osm_route=[0, 1, 2],
        active=True,
        fcd_sampling_period=timedelta(seconds=5),
        status="",
        vehicle_type="truck",
    )
    segments = [
        Segment(0, 1, LengthMeters(1000.0), SpeedKph(90.0), 1),
        Segment(1, 2, LengthMeters(1000.0), SpeedKph(90.0), 1),
    ]
    routing_map = SimpleNamespace(
        osm_route_to_py_segments=lambda nodes: segments[:len(nodes) - 1],
        current_network={0: {1: {0: {"current_speed": 90, "highway": "primary"}}}},
        network=SimpleNamespace(nodes={}),
    )
    global_view = HeterogeneousGlobalView()
    global_view.register_vehicles([vehicle])
    queues_manager = QueuesManager()

    fcds, moved = advance_vehicles_with_queues(
        [vehicle],
        current_time,
        global_view,
        routing_map,
        queues_manager,
        timedelta(seconds=0),
        movement_model=HeterogeneousMovementModel(
            global_view,
            routing_map,
            timedelta(seconds=0),
        ),
    )

    assert moved is True
    assert fcds
    assert not hasattr(fcds[0], "vehicle_type")
    assert queues_manager.to_be_added == []
    assert queues_manager.to_be_removed == []

    for fcd in fcds:
        global_view.add(fcd)
    total_pcu, _, _ = global_view.load_ahead(
        fcds[-1].datetime,
        fcds[-1].segment.id,
        vehicle_offset_m=-1,
    )
    assert total_pcu == DEFAULT_VEHICLE_CLASSES["truck"].pcu
