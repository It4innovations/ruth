import pytest
from collections import deque
from datetime import timedelta

from ruth.simulator.queues import QueuesManager
from ruth.vehicle import Vehicle
from ruth.data.segment import LengthMeters


@pytest.fixture
def queues_manager():
    return QueuesManager()


@pytest.fixture
def sample_vehicle():
    return Vehicle(
        id=1,
        time_offset=timedelta(seconds=0),
        frequency=timedelta(seconds=10),
        start_index=0,
        start_distance_offset=LengthMeters(10.0),
        origin_node=0,
        dest_node=2,
        osm_route=[0, 1, 2],
        active=True,
        fcd_sampling_period=timedelta(seconds=5),
        status=""
    )


@pytest.fixture
def multiple_vehicles():
    return [
        Vehicle(
            id=i,
            time_offset=timedelta(seconds=0),
            frequency=timedelta(seconds=10),
            start_index=0,
            start_distance_offset=LengthMeters(10.0),
            origin_node=0,
            dest_node=3,
            osm_route=[0, 1, 2, 3],
            active=True,
            fcd_sampling_period=timedelta(seconds=5),
            status=""
        )
        for i in range(5)
    ]


def test_initialization(queues_manager):
    """Test that QueuesManager initializes with empty data structures."""
    assert isinstance(queues_manager.queues, dict)
    assert len(queues_manager.queues) == 0
    assert isinstance(queues_manager.to_be_added, list)
    assert len(queues_manager.to_be_added) == 0
    assert isinstance(queues_manager.to_be_removed, list)
    assert len(queues_manager.to_be_removed) == 0


def test_add_single_vehicle(queues_manager, sample_vehicle):
    """Test adding a single vehicle to queue."""
    queues_manager.add_to_queue(sample_vehicle)

    assert len(queues_manager.to_be_added) == 1
    assert queues_manager.to_be_added[0] == (
        sample_vehicle.id,
        sample_vehicle.current_node,
        sample_vehicle.next_node
    )


def test_add_multiple_vehicles_same_segment(queues_manager, multiple_vehicles):
    """Test adding multiple vehicles to the same queue."""
    for vehicle in multiple_vehicles:
        queues_manager.add_to_queue(vehicle)

    assert len(queues_manager.to_be_added) == len(multiple_vehicles)
    vehicle_ids = [item[0] for item in queues_manager.to_be_added]
    assert set(vehicle_ids) == {v.id for v in multiple_vehicles}


def test_add_vehicle_different_segments(queues_manager):
    """Test adding vehicles to different segments."""
    vehicle1 = Vehicle(
        id=1, time_offset=timedelta(seconds=0), frequency=timedelta(seconds=10),
        start_index=0, start_distance_offset=LengthMeters(0.0),
        origin_node=0, dest_node=2, osm_route=[0, 1, 2],
        active=True, fcd_sampling_period=timedelta(seconds=5), status=""
    )
    vehicle2 = Vehicle(
        id=2, time_offset=timedelta(seconds=0), frequency=timedelta(seconds=10),
        start_index=1, start_distance_offset=LengthMeters(0.0),
        origin_node=0, dest_node=3, osm_route=[0, 1, 2, 3],
        active=True, fcd_sampling_period=timedelta(seconds=5), status=""
    )

    queues_manager.add_to_queue(vehicle1)
    queues_manager.add_to_queue(vehicle2)

    assert len(queues_manager.to_be_added) == 2
    assert queues_manager.to_be_added[0] == (1, 0, 1)
    assert queues_manager.to_be_added[1] == (2, 1, 2)


def test_add_does_not_immediately_update_queues(queues_manager, sample_vehicle):
    """Test that add_to_queue doesn't immediately update the queues dict."""
    queues_manager.add_to_queue(sample_vehicle)

    assert len(queues_manager.queues) == 0
    assert len(queues_manager.to_be_added) == 1


def test_remove_vehicle(queues_manager, sample_vehicle):
    """Test removing a vehicle from queue."""
    queues_manager.remove_vehicle(sample_vehicle, 0, 1)

    assert len(queues_manager.to_be_removed) == 1
    assert queues_manager.to_be_removed[0] == (sample_vehicle.id, 0, 1)


def test_remove_multiple_vehicles(queues_manager, multiple_vehicles):
    """Test removing multiple vehicles."""
    for vehicle in multiple_vehicles:
        queues_manager.remove_vehicle(vehicle, 0, 1)

    assert len(queues_manager.to_be_removed) == len(multiple_vehicles)


def test_remove_does_not_immediately_update_queues(queues_manager, sample_vehicle):
    """Test that remove_vehicle doesn't immediately update the queues dict."""
    queues_manager.add_to_queue(sample_vehicle)
    queues_manager.batch_update()

    queues_manager.remove_vehicle(sample_vehicle, 0, 1)

    assert sample_vehicle.id in queues_manager.queues[(0, 1)]
    assert len(queues_manager.to_be_removed) == 1


def test_remove_inactive_vehicle_in_to_be_removed(queues_manager, sample_vehicle):
    """Test that remove_inactive_vehicle handles vehicle already in to_be_removed."""
    queues_manager.to_be_removed.append((sample_vehicle.id, 0, 1))
    queues_manager.remove_inactive_vehicle(sample_vehicle)


def test_remove_inactive_vehicle_not_in_queues(queues_manager, sample_vehicle):
    """Test removing inactive vehicle not in any queue."""
    queues_manager.remove_inactive_vehicle(sample_vehicle)


def test_remove_inactive_vehicle_in_queue_raises_error(queues_manager, sample_vehicle):
    """Test that remove_inactive_vehicle raises error if vehicle is in queue but not in to_be_removed."""
    queues_manager.queues[(0, 1)] = deque([sample_vehicle.id])

    with pytest.raises(ValueError, match="Vehicle not found in to_be_removed"):
        queues_manager.remove_inactive_vehicle(sample_vehicle)


def test_batch_update_adds_vehicles(queues_manager, sample_vehicle):
    """Test that batch_update processes to_be_added."""
    queues_manager.add_to_queue(sample_vehicle)
    queues_manager.batch_update()

    queue_key = (sample_vehicle.current_node, sample_vehicle.next_node)
    assert queue_key in queues_manager.queues
    assert sample_vehicle.id in queues_manager.queues[queue_key]
    assert len(queues_manager.to_be_added) == 0


def test_batch_update_removes_vehicles(queues_manager, sample_vehicle):
    """Test that batch_update processes to_be_removed."""
    queues_manager.add_to_queue(sample_vehicle)
    queues_manager.batch_update()

    queues_manager.remove_vehicle(sample_vehicle, 0, 1)
    queues_manager.batch_update()

    assert len(queues_manager.queues[(0, 1)]) == 0
    assert len(queues_manager.to_be_removed) == 0


def test_batch_update_maintains_fifo_order(queues_manager, multiple_vehicles):
    """Test that batch_update maintains FIFO order."""
    for vehicle in multiple_vehicles:
        queues_manager.add_to_queue(vehicle)

    queues_manager.batch_update()

    queue = queues_manager.queues[(0, 1)]
    for i, vehicle_id in enumerate(queue):
        assert vehicle_id == multiple_vehicles[i].id


def test_batch_update_removes_from_front(queues_manager, multiple_vehicles):
    """Test that vehicles are removed from the front of the queue (FIFO)."""
    for vehicle in multiple_vehicles:
        queues_manager.add_to_queue(vehicle)
    queues_manager.batch_update()

    queues_manager.remove_vehicle(multiple_vehicles[0], 0, 1)
    queues_manager.batch_update()

    queue = queues_manager.queues[(0, 1)]
    assert multiple_vehicles[0].id not in queue
    assert len(queue) == len(multiple_vehicles) - 1
    for i, vehicle_id in enumerate(queue):
        assert vehicle_id == multiple_vehicles[i + 1].id


def test_batch_update_empty_operations(queues_manager):
    """Test batch_update with no pending operations."""
    queues_manager.batch_update()

    assert len(queues_manager.to_be_added) == 0
    assert len(queues_manager.to_be_removed) == 0
    assert len(queues_manager.queues) == 0


def test_batch_update_mixed_operations(queues_manager, multiple_vehicles):
    """Test batch_update with both add and remove operations."""
    for vehicle in multiple_vehicles[:3]:
        queues_manager.add_to_queue(vehicle)
    queues_manager.batch_update()

    queues_manager.remove_vehicle(multiple_vehicles[0], 0, 1)
    for vehicle in multiple_vehicles[3:]:
        queues_manager.add_to_queue(vehicle)

    queues_manager.batch_update()

    queue = queues_manager.queues[(0, 1)]
    assert multiple_vehicles[0].id not in queue
    assert len(queue) == len(multiple_vehicles) - 1
    expected_ids = [v.id for v in multiple_vehicles[1:]]
    actual_ids = list(queue)
    assert actual_ids == expected_ids


def test_batch_update_assertion_on_empty_queue(queues_manager, sample_vehicle):
    """Test that removing from empty queue raises assertion error."""
    queues_manager.remove_vehicle(sample_vehicle, 0, 1)

    with pytest.raises(AssertionError):
        queues_manager.batch_update()


def test_batch_update_assertion_on_wrong_vehicle(queues_manager, multiple_vehicles):
    """Test that removing wrong vehicle from queue raises assertion error."""
    queues_manager.add_to_queue(multiple_vehicles[0])
    queues_manager.batch_update()

    queues_manager.remove_vehicle(multiple_vehicles[1], 0, 1)

    with pytest.raises(AssertionError):
        queues_manager.batch_update()


def test_multiple_separate_queues(queues_manager):
    """Test managing vehicles in different queue segments."""
    vehicle1 = Vehicle(
        id=1, time_offset=timedelta(seconds=0), frequency=timedelta(seconds=10),
        start_index=0, start_distance_offset=LengthMeters(0.0),
        origin_node=0, dest_node=3, osm_route=[0, 1, 2, 3],
        active=True, fcd_sampling_period=timedelta(seconds=5), status=""
    )
    vehicle2 = Vehicle(
        id=2, time_offset=timedelta(seconds=0), frequency=timedelta(seconds=10),
        start_index=1, start_distance_offset=LengthMeters(0.0),
        origin_node=0, dest_node=3, osm_route=[0, 1, 2, 3],
        active=True, fcd_sampling_period=timedelta(seconds=5), status=""
    )

    queues_manager.add_to_queue(vehicle1)
    queues_manager.add_to_queue(vehicle2)
    queues_manager.batch_update()

    assert len(queues_manager.queues) == 2
    assert (0, 1) in queues_manager.queues
    assert (1, 2) in queues_manager.queues
    assert vehicle1.id in queues_manager.queues[(0, 1)]
    assert vehicle2.id in queues_manager.queues[(1, 2)]


def test_operations_on_different_queues(queues_manager, multiple_vehicles):
    """Test that operations on one queue don't affect others."""
    for i, vehicle in enumerate(multiple_vehicles[:3]):
        vehicle.start_index = 0
        queues_manager.add_to_queue(vehicle)

    for i, vehicle in enumerate(multiple_vehicles[3:]):
        vehicle.start_index = 1
        queues_manager.add_to_queue(vehicle)

    queues_manager.batch_update()

    queues_manager.remove_vehicle(multiple_vehicles[0], 0, 1)
    queues_manager.batch_update()

    assert len(queues_manager.queues[(0, 1)]) == 2
    assert len(queues_manager.queues[(1, 2)]) == 2


def test_defaultdict_creates_new_queue(queues_manager):
    """Test that accessing non-existent queue creates it via defaultdict."""
    queue = queues_manager.queues[(99, 100)]
    assert isinstance(queue, deque)
    assert len(queue) == 0


def test_same_vehicle_added_multiple_times(queues_manager, sample_vehicle):
    """Test adding same vehicle multiple times before batch_update."""
    queues_manager.add_to_queue(sample_vehicle)
    queues_manager.add_to_queue(sample_vehicle)

    assert len(queues_manager.to_be_added) == 2

    queues_manager.batch_update()

    queue = queues_manager.queues[(0, 1)]
    assert len(queue) == 2
    assert list(queue) == [sample_vehicle.id, sample_vehicle.id]


def test_vehicle_with_none_next_node(queues_manager):
    """Test handling vehicle at end of route (next_node is None)."""
    vehicle = Vehicle(
        id=1, time_offset=timedelta(seconds=0), frequency=timedelta(seconds=10),
        start_index=1,
        start_distance_offset=LengthMeters(0.0),
        origin_node=0, dest_node=1, osm_route=[0, 1],
        active=True, fcd_sampling_period=timedelta(seconds=5), status=""
    )

    assert vehicle.next_node is None

    queues_manager.add_to_queue(vehicle)
    queues_manager.batch_update()

    assert (1, None) in queues_manager.queues


def test_clear_after_batch_update(queues_manager, multiple_vehicles):
    """Test that to_be_added and to_be_removed are cleared after batch_update."""
    for vehicle in multiple_vehicles:
        queues_manager.add_to_queue(vehicle)

    assert len(queues_manager.to_be_added) > 0

    queues_manager.batch_update()

    assert len(queues_manager.to_be_added) == 0

    for vehicle in multiple_vehicles:
        queues_manager.remove_vehicle(vehicle, 0, 1)

    assert len(queues_manager.to_be_removed) > 0

    queues_manager.batch_update()

    assert len(queues_manager.to_be_removed) == 0


def test_vehicle_moving_through_segments(queues_manager):
    """Test simulating a vehicle moving through multiple segments."""
    vehicle = Vehicle(
        id=1, time_offset=timedelta(seconds=0), frequency=timedelta(seconds=10),
        start_index=0, start_distance_offset=LengthMeters(0.0),
        origin_node=0, dest_node=3, osm_route=[0, 1, 2, 3],
        active=True, fcd_sampling_period=timedelta(seconds=5), status=""
    )

    queues_manager.add_to_queue(vehicle)
    queues_manager.batch_update()
    assert vehicle.id in queues_manager.queues[(0, 1)]

    queues_manager.remove_vehicle(vehicle, 0, 1)
    vehicle.start_index = 1
    queues_manager.add_to_queue(vehicle)
    queues_manager.batch_update()

    assert vehicle.id not in queues_manager.queues[(0, 1)]
    assert vehicle.id in queues_manager.queues[(1, 2)]

    queues_manager.remove_vehicle(vehicle, 1, 2)
    vehicle.start_index = 2
    queues_manager.add_to_queue(vehicle)
    queues_manager.batch_update()

    assert vehicle.id not in queues_manager.queues[(1, 2)]
    assert vehicle.id in queues_manager.queues[(2, 3)]


def test_multiple_vehicles_queue_progression(queues_manager, multiple_vehicles):
    """Test multiple vehicles progressing through a queue."""
    for vehicle in multiple_vehicles:
        queues_manager.add_to_queue(vehicle)
    queues_manager.batch_update()

    queue = queues_manager.queues[(0, 1)]
    assert len(queue) == len(multiple_vehicles)

    queues_manager.remove_vehicle(multiple_vehicles[0], 0, 1)
    queues_manager.batch_update()

    assert len(queues_manager.queues[(0, 1)]) == len(multiple_vehicles) - 1

    queues_manager.remove_vehicle(multiple_vehicles[1], 0, 1)
    queues_manager.batch_update()

    assert len(queues_manager.queues[(0, 1)]) == len(multiple_vehicles) - 2


def test_congestion_buildup_and_clearing(queues_manager, multiple_vehicles):
    """Test simulating traffic congestion buildup and clearing."""
    for i, vehicle in enumerate(multiple_vehicles):
        queues_manager.add_to_queue(vehicle)
        if i % 2 == 0:
            queues_manager.batch_update()

    queues_manager.batch_update()

    queue = queues_manager.queues[(0, 1)]
    max_queue_length = len(queue)
    assert max_queue_length == len(multiple_vehicles)

    for vehicle in multiple_vehicles:
        queues_manager.remove_vehicle(vehicle, 0, 1)
        queues_manager.batch_update()

        queue = queues_manager.queues[(0, 1)]
        assert len(queue) < max_queue_length
        max_queue_length = len(queue)

    assert len(queues_manager.queues[(0, 1)]) == 0
