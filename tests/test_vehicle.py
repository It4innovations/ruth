import pytest
from datetime import timedelta
from unittest.mock import MagicMock

from ruth.data.segment import LengthMeters, TravelTime
from ruth.data.map import Map, BBox
from ruth.vehicle import (
    Vehicle,
    VehicleAlternatives,
    VehicleRouteSelection,
    CurrentTravelTime,
    set_vehicle_behavior,
)


@pytest.fixture
def sample_vehicle():
    """Create a basic test vehicle with a real OSM route."""
    return Vehicle(
        id=1,
        time_offset=timedelta(seconds=0),
        frequency=timedelta(seconds=10),
        start_index=0,
        start_distance_offset=LengthMeters(0.0),
        origin_node=172512,
        dest_node=172514,
        osm_route=[172512, 300107261, 3896663846, 172513, 172514],
        active=True,
        fcd_sampling_period=timedelta(seconds=5),
        status=""
    )


@pytest.fixture
def inactive_vehicle():
    """Create an inactive vehicle with a real OSM route."""
    return Vehicle(
        id=2,
        time_offset=timedelta(seconds=5),
        frequency=timedelta(seconds=10),
        start_index=0,
        start_distance_offset=LengthMeters(0.0),
        origin_node=172512,
        dest_node=172513,
        osm_route=[172512, 300107261, 3896663846, 172513],
        active=False,
        fcd_sampling_period=timedelta(seconds=5),
        status=""
    )


@pytest.fixture(scope='module')
def routing_map():
    """Create a real routing map for testing."""
    bbox = BBox(50.16568920000002, 14.321441000000016, 50.020240399999985, 14.592499399999983)
    routing_map = Map(bbox=bbox, download_date='2024-01-10T00:00:00', with_speeds=True)
    return routing_map



def test_vehicle_current_node(sample_vehicle):
    """Test getting the current node of the vehicle."""
    # osm_route = [172512, 300107261, 3896663846, 172513, 172514]
    assert sample_vehicle.current_node == sample_vehicle.osm_route[0]

    sample_vehicle.start_index = 1
    assert sample_vehicle.current_node == sample_vehicle.osm_route[1]

    sample_vehicle.start_index = 2
    assert sample_vehicle.current_node == sample_vehicle.osm_route[2]


def test_vehicle_next_node(sample_vehicle):
    """Test getting the next node of the vehicle."""
    # osm_route = [172512, 300107261, 3896663846, 172513, 172514]
    assert sample_vehicle.next_node == sample_vehicle.osm_route[1]

    sample_vehicle.start_index = 1
    assert sample_vehicle.next_node == sample_vehicle.osm_route[2]

    sample_vehicle.start_index = 2
    assert sample_vehicle.next_node == sample_vehicle.osm_route[3]

    sample_vehicle.start_index = 3
    assert sample_vehicle.next_node == sample_vehicle.osm_route[4]

    sample_vehicle.start_index = 4
    assert sample_vehicle.next_node is None


def test_vehicle_segment_position(sample_vehicle):
    """Test getting and setting vehicle segment position."""
    position = sample_vehicle.segment_position
    assert position.index == 0
    assert position.position == LengthMeters(0.0)

    from ruth.data.segment import SegmentPosition
    new_position = SegmentPosition(index=1, position=LengthMeters(100.0))
    sample_vehicle.set_position(new_position)

    assert sample_vehicle.start_index == 1
    assert sample_vehicle.start_distance_offset == 100.0


def test_vehicle_equality():
    """Test vehicle equality based on ID."""
    v1 = Vehicle(
        id=1, time_offset=timedelta(seconds=0), frequency=timedelta(seconds=10),
        start_index=0, start_distance_offset=LengthMeters(0.0),
        origin_node=0, dest_node=2, osm_route=[0, 1, 2],
        active=True, fcd_sampling_period=timedelta(seconds=5), status=""
    )
    v2 = Vehicle(
        id=1, time_offset=timedelta(seconds=5), frequency=timedelta(seconds=10),
        start_index=1, start_distance_offset=LengthMeters(50.0),
        origin_node=0, dest_node=2, osm_route=[0, 1, 2],
        active=False, fcd_sampling_period=timedelta(seconds=5), status=""
    )
    v3 = Vehicle(
        id=2, time_offset=timedelta(seconds=0), frequency=timedelta(seconds=10),
        start_index=0, start_distance_offset=LengthMeters(0.0),
        origin_node=0, dest_node=2, osm_route=[0, 1, 2],
        active=True, fcd_sampling_period=timedelta(seconds=5), status=""
    )

    assert v1 == v2  # Same ID
    assert v1 != v3  # Different ID


def test_next_routing_od_nodes(sample_vehicle):
    """Test next routing origin-destination nodes."""
    next_od = sample_vehicle.next_routing_od_nodes
    assert next_od == (sample_vehicle.osm_route[0], sample_vehicle.osm_route[-1])

    sample_vehicle.start_index = 1
    next_od = sample_vehicle.next_routing_od_nodes
    assert next_od == (sample_vehicle.osm_route[1], sample_vehicle.osm_route[-1])


def test_next_routing_start(sample_vehicle):
    """Test next routing start point."""
    start = sample_vehicle.next_routing_start
    assert start.node == sample_vehicle.osm_route[0]
    assert start.index == 0

    sample_vehicle.start_index = 1
    start = sample_vehicle.next_routing_start
    assert start.node == sample_vehicle.osm_route[1]
    assert start.index == 1

    # When vehicle has traveled some distance on current segment
    sample_vehicle.start_index = 1
    sample_vehicle.start_distance_offset = LengthMeters(50.0)
    start = sample_vehicle.next_routing_start
    assert start.node == sample_vehicle.osm_route[2]
    assert start.index == 2


def test_k_shortest_paths(sample_vehicle, routing_map):
    """Test computing k-shortest paths."""
    number_of_paths = 5
    paths = sample_vehicle.k_shortest_paths(number_of_paths, routing_map)

    # Verify we got paths and they are valid
    assert paths is not None
    assert len(paths) > 0
    for path in paths:
        assert path[0] == sample_vehicle.osm_route[0]
        assert path[-1] == sample_vehicle.osm_route[-1]

    # check total length of paths is not decreasing
    lengths = []
    for path in paths:
        length = 0.0
        for i in range(len(path) - 1):
            segment = routing_map.get_osm_segment(path[i], path[i + 1])
            length += segment.length
        lengths.append(length)
    assert all(lengths[i] <= lengths[i + 1] for i in range(len(lengths) - 1))



def test_k_shortest_paths_node_not_found(sample_vehicle, routing_map):
    """Test k_shortest_paths when node is not found."""
    bad_vehicle = Vehicle(
        id=99, time_offset=timedelta(seconds=0), frequency=timedelta(seconds=10),
        start_index=0, start_distance_offset=LengthMeters(0.0),
        origin_node=999999999, dest_node=888888888,
        osm_route=[999999999, 888888888],
        active=True, fcd_sampling_period=timedelta(seconds=5), status=""
    )

    paths = bad_vehicle.k_shortest_paths(2, routing_map)
    assert paths is None


def test_k_shortest_paths_different_start_index(sample_vehicle, routing_map):
    """Test k_shortest_paths from different starting indices."""
    sample_vehicle.start_index = 1

    paths = sample_vehicle.k_shortest_paths(2, routing_map)

    # Should get paths starting from osm_route[1]
    assert paths is not None
    assert len(paths) > 0
    for path in paths:
        assert path[0] == sample_vehicle.osm_route[1]
        assert path[-1] == sample_vehicle.osm_route[-1]


def test_k_fastest_paths(sample_vehicle, routing_map):
    """Test computing k-fastest paths."""
    paths = sample_vehicle.k_fastest_paths(2, routing_map)

    assert paths is not None
    assert len(paths) > 0
    for path in paths:
        assert path[0] == sample_vehicle.osm_route[0]
        assert path[-1] == sample_vehicle.osm_route[-1]

    # check travel time of each path is non-decreasing
    travel_times = []
    for path in paths:
        travel_time = routing_map.get_path_travel_time(path)
        travel_times.append(travel_time)
    assert all(earlier <= later for earlier, later in zip(travel_times, travel_times[1:]))

def test_k_fastest_paths_node_not_found(sample_vehicle, routing_map):
    """Test k_fastest_paths when node is not found."""
    # Create a vehicle with non-existent nodes
    bad_vehicle = Vehicle(
        id=99, time_offset=timedelta(seconds=0), frequency=timedelta(seconds=10),
        start_index=0, start_distance_offset=LengthMeters(0.0),
        origin_node=999999999, dest_node=888888888,
        osm_route=[999999999, 888888888],
        active=True, fcd_sampling_period=timedelta(seconds=5), status=""
    )

    paths = bad_vehicle.k_fastest_paths(2, routing_map)
    assert paths is None


def test_k_fastest_paths_different_start_index(sample_vehicle, routing_map):
    """Test k_fastest_paths from different starting indices."""
    sample_vehicle.start_index = 1

    paths = sample_vehicle.k_fastest_paths(2, routing_map)

    # Should get paths starting from osm_route[1]
    assert paths is not None
    assert len(paths) > 0
    for path in paths:
        assert path[0] == sample_vehicle.osm_route[1]
        assert path[-1] == sample_vehicle.osm_route[-1]


def test_map_id_property(sample_vehicle):
    """Test map_id property."""
    assert sample_vehicle.map_id == -1

    sample_vehicle.set_current_travel_time(100.0, map_id=2)
    assert sample_vehicle.map_id == 2

    assert sample_vehicle.current_travel_time is not None
    assert sample_vehicle.current_travel_time.travel_time == 100.0
    assert sample_vehicle.current_travel_time.start_segment_index == 1


def test_get_travel_time_limit(sample_vehicle):
    """Test getting travel time limit."""
    sample_vehicle.set_current_travel_time(100.0, map_id=1)

    limit = sample_vehicle.get_travel_time_limit(1, 0.5)
    assert limit == 50.0  # 100 * (1 - 0.5)


def test_get_travel_time_limit_wrong_map_id(sample_vehicle):
    """Test getting travel time limit with wrong map ID."""
    sample_vehicle.set_current_travel_time(100.0, map_id=1)

    with pytest.raises(AssertionError):
        sample_vehicle.get_travel_time_limit(2, 0.5)


def test_get_followup_route(sample_vehicle):
    """Test getting the followup route."""
    route = sample_vehicle.get_followup_route()
    assert route == sample_vehicle.osm_route

    sample_vehicle.start_index = 1
    route = sample_vehicle.get_followup_route()
    assert route == sample_vehicle.osm_route[1:]


def test_subtract_from_travel_time(sample_vehicle):
    """Test subtracting from travel time."""
    sample_vehicle.set_current_travel_time(100.0, map_id=1)

    sample_vehicle.subtract_from_travel_time(25.0, node_from=1)

    assert sample_vehicle.current_travel_time.travel_time == 75.0
    assert sample_vehicle.current_travel_time.start_segment_index == 1


def test_update_followup_route_dijkstra_shortest(sample_vehicle, routing_map):
    """Test updating followup route with DIJKSTRA_SHORTEST alternatives."""
    sample_vehicle.alternatives = VehicleAlternatives.DIJKSTRA_SHORTEST
    suggested_route_with_time = ([sample_vehicle.osm_route[0], 999, sample_vehicle.osm_route[-1]], None)

    sample_vehicle.update_followup_route(suggested_route_with_time, routing_map)

    # first_part is osm_route[:0] = [], so result is [] + [172512, 999, 172514]
    assert sample_vehicle.osm_route == [sample_vehicle.osm_route[0], 999, sample_vehicle.osm_route[-1]]


def test_update_followup_route_no_travel_time_limit(sample_vehicle, routing_map):
    """Test updating followup route without travel time limit."""
    sample_vehicle.set_current_travel_time(100.0, map_id=1)

    suggested_route_with_time = ([sample_vehicle.osm_route[0], 888, sample_vehicle.osm_route[-1]], None)
    sample_vehicle.update_followup_route(suggested_route_with_time, routing_map, travel_time_limit_perc=None)
    assert sample_vehicle.osm_route == [sample_vehicle.osm_route[0], 888, sample_vehicle.osm_route[-1]]


def test_update_followup_route_with_travel_time_limit(sample_vehicle, routing_map):
    """Test updating followup route with travel time limit constraint."""
    sample_vehicle.start_index = 0

    sample_vehicle.set_current_travel_time(100.0, map_id=1)
    sample_vehicle.alternatives = VehicleAlternatives.DIJKSTRA_FASTEST

    suggested_route_with_time = ([sample_vehicle.osm_route[0], 777, sample_vehicle.osm_route[-1]], TravelTime(40.0))
    sample_vehicle.update_followup_route(suggested_route_with_time, routing_map, travel_time_limit_perc=0.5)
    # 40.0 < 50.0 (100 * 0.5), so route should be updated
    assert sample_vehicle.osm_route == [sample_vehicle.osm_route[0], 777, sample_vehicle.osm_route[-1]]


def test_update_followup_route_exceeds_time_limit(sample_vehicle, routing_map):
    """Test updating followup route when suggested route exceeds time limit."""
    sample_vehicle.start_index = 0
    sample_vehicle.osm_route = [172512, 300107261, 3896663846, 172513, 172514]

    sample_vehicle.set_current_travel_time(100.0, map_id=1)
    sample_vehicle.alternatives = VehicleAlternatives.DIJKSTRA_FASTEST
    # Route must end with dest_node (172514)
    suggested_route_with_time = ([sample_vehicle.osm_route[0], 666, sample_vehicle.osm_route[-1]], TravelTime(60.0))

    original_route = sample_vehicle.osm_route.copy()
    sample_vehicle.update_followup_route(suggested_route_with_time, routing_map, travel_time_limit_perc=0.5)

    # 60.0 > 50.0 (100 * 0.5), so route should NOT be updated
    assert sample_vehicle.osm_route == original_route

def test_is_at_the_end_of_segment(sample_vehicle, routing_map):
    """Test checking if vehicle is at the end of segment."""
    sample_vehicle.start_distance_offset = LengthMeters(0.0)
    assert sample_vehicle.is_at_the_end_of_segment(routing_map) is False

    segment = routing_map.get_osm_segment(sample_vehicle.osm_route[0], sample_vehicle.osm_route[1])
    sample_vehicle.start_distance_offset = LengthMeters(segment.length)
    assert sample_vehicle.is_at_the_end_of_segment(routing_map) is True


def test_has_next_segment_closed(sample_vehicle, routing_map):
    result = sample_vehicle.has_next_segment_closed(routing_map)
    assert result is False

    # set speed for the next segment to 0 to simulate closure
    node_from = sample_vehicle.osm_route[1]
    node_to = sample_vehicle.osm_route[2]
    routing_map.current_network[node_from][node_to]['speed_kph'] = 0.0

    result = sample_vehicle.has_next_segment_closed(routing_map)
    assert result is True


def test_vehicle_frequency_caching(sample_vehicle):
    """Test that frequency is cached in seconds."""
    assert sample_vehicle._frequency_seconds == 10

    sample_vehicle.frequency = timedelta(seconds=20)
    assert sample_vehicle._frequency_seconds == 20


def test_set_vehicle_behavior_basic():
    """Test setting vehicle behavior with basic ratios."""
    vehicles = [
        Vehicle(
            id=i, time_offset=timedelta(seconds=0), frequency=timedelta(seconds=10),
            start_index=0, start_distance_offset=LengthMeters(0.0),
            origin_node=0, dest_node=2, osm_route=[0, 1, 2],
            active=True, fcd_sampling_period=timedelta(seconds=5), status=""
        )
        for i in range(10)
    ]

    alternatives_ratio = [0.5, 0.5, 0.0, 0.0]
    route_selection_ratio = [0.5, 0.5, 0.0, 0.0]

    set_vehicle_behavior(vehicles, alternatives_ratio, route_selection_ratio)

    # Check that vehicles have been assigned alternatives
    default_count = sum(1 for v in vehicles if v.alternatives == VehicleAlternatives.DEFAULT)
    assert default_count == 5

    # Check that vehicles have been assigned route selection
    no_alt_count = sum(1 for v in vehicles if v.route_selection == VehicleRouteSelection.NO_ALTERNATIVE)
    assert no_alt_count == 5


def test_set_vehicle_behavior_all_alternatives():
    """Test setting vehicle behavior with all alternatives."""
    vehicles = [
        Vehicle(
            id=i, time_offset=timedelta(seconds=0), frequency=timedelta(seconds=10),
            start_index=0, start_distance_offset=LengthMeters(0.0),
            origin_node=0, dest_node=2, osm_route=[0, 1, 2],
            active=True, fcd_sampling_period=timedelta(seconds=5), status=""
        )
        for i in range(4)
    ]

    alternatives_ratio = [0.25, 0.25, 0.25, 0.25]
    route_selection_ratio = [0.25, 0.25, 0.25, 0.25]

    set_vehicle_behavior(vehicles, alternatives_ratio, route_selection_ratio)

    # Each alternative should be assigned once
    for alt in VehicleAlternatives:
        alt_count = sum(1 for v in vehicles if v.alternatives == alt)
        assert alt_count == 1


def test_set_vehicle_behavior_default_only():
    """Test setting vehicle behavior with only default alternatives."""
    vehicles = [
        Vehicle(
            id=i, time_offset=timedelta(seconds=0), frequency=timedelta(seconds=10),
            start_index=0, start_distance_offset=LengthMeters(0.0),
            origin_node=0, dest_node=2, osm_route=[0, 1, 2],
            active=True, fcd_sampling_period=timedelta(seconds=5), status=""
        )
        for i in range(5)
    ]

    alternatives_ratio = [1.0, 0.0, 0.0, 0.0]
    route_selection_ratio = [1.0, 0.0, 0.0, 0.0]

    set_vehicle_behavior(vehicles, alternatives_ratio, route_selection_ratio)

    # All should have DEFAULT
    assert all(v.alternatives == VehicleAlternatives.DEFAULT for v in vehicles)
    assert all(v.route_selection == VehicleRouteSelection.NO_ALTERNATIVE for v in vehicles)


def test_set_vehicle_behavior_mismatched_default_no_alternative():
    """Test that mismatched DEFAULT/NO_ALTERNATIVE ratios raise error."""
    vehicles = [
        Vehicle(
            id=i, time_offset=timedelta(seconds=0), frequency=timedelta(seconds=10),
            start_index=0, start_distance_offset=LengthMeters(0.0),
            origin_node=0, dest_node=2, osm_route=[0, 1, 2],
            active=True, fcd_sampling_period=timedelta(seconds=5), status=""
        )
        for i in range(10)
    ]

    alternatives_ratio = [0.6, 0.4, 0.0, 0.0]
    route_selection_ratio = [0.5, 0.5, 0.0, 0.0]

    with pytest.raises(ValueError, match="must be equal"):
        set_vehicle_behavior(vehicles, alternatives_ratio, route_selection_ratio)


def test_set_vehicle_behavior_invalid_ratios():
    """Test that invalid ratios raise assertion error."""
    vehicles = [
        Vehicle(
            id=i, time_offset=timedelta(seconds=0), frequency=timedelta(seconds=10),
            start_index=0, start_distance_offset=LengthMeters(0.0),
            origin_node=0, dest_node=2, osm_route=[0, 1, 2],
            active=True, fcd_sampling_period=timedelta(seconds=5), status=""
        )
        for i in range(10)
    ]

    alternatives_ratio = [0.5, 0.3, 0.0, 0.0]  # Sum = 0.8, not 1.0
    route_selection_ratio = [0.5, 0.3, 0.0, 0.0]

    with pytest.raises(AssertionError):
        set_vehicle_behavior(vehicles, alternatives_ratio, route_selection_ratio)


def test_vehicle_pickle_compatibility(sample_vehicle):
    import pickle

    pickled = pickle.dumps(sample_vehicle)
    unpickled = pickle.loads(pickled)

    for attr in vars(sample_vehicle):
        assert getattr(unpickled, attr) == getattr(sample_vehicle, attr)
