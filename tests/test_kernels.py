from cmath import isclose
from time import sleep

import pytest

from datetime import datetime, timedelta

from ruth.data.segment import LengthMeters, SpeedKph, speed_kph_to_mps
from ruth.simulator.kernels import FastestPathsAlternatives, ZeroMQDistributedAlternatives
from ruth.tools.simulator import CommonArgs, prepare_simulator, AlternativesRatio, RouteSelectionRatio, ZeroMqContext

from ruth.vehicle import Vehicle

expected_alt = [25664661, 27349583, 27350859, 1421692706, 302546398, 292456769, 1839597743, 1467045348, 21713539,
                2158126162, 386387776, 25665323, 236788246, 32438879, 4823824959, 21673412, 29527230, 344481246,
                29527229, 244011412, 11267251949, 1483392086, 21311889, 25972978, 170348063, 25972980, 25972985,
                306614568, 9254639009, 29403188, 305832258]
expected_travel_time = 345


@pytest.fixture
def setup_vehicle():
    vehicle = Vehicle(
        id=1,
        time_offset=timedelta(seconds=120),
        frequency=timedelta(seconds=10),
        start_index=0,
        start_distance_offset=LengthMeters(0.0),
        origin_node=25664661,
        dest_node=305832258,
        osm_route=[25664661, 27349583, 27350859, 1421692706, 302546398, 292456769, 1839597743, 1467045348, 21713539,
                   2158126162, 386387776, 25665323, 236788246, 32438879, 4823824959, 21673412, 29527230, 344481246,
                   29527229, 29527228, 11267251949, 1483392086, 21311889, 25972978, 170348063, 25972980, 25972985,
                   306614568, 9254639009, 29403188, 305832258],
        active=True,
        fcd_sampling_period=timedelta(seconds=5),
        status=""
    )
    return vehicle


@pytest.fixture
def setup_simulator():
    common_args = CommonArgs(
        task_id="test",
        departure_time=datetime(2021, 1, 1, 0, 0, 0),
        round_frequency=timedelta(seconds=5),
        k_alternatives=1,
        map_update_freq=timedelta(seconds=1),
        los_vehicles_tolerance=timedelta(seconds=0),
    )

    alternatives_ratio = AlternativesRatio(
        default=0.0,
        dijkstra_fastest=0.0,
        dijkstra_shortest=1.0,
        plateau_fastest=0.0
    )

    route_selection_ratio = RouteSelectionRatio(
        no_alternative=0.0,
        first=1.0,
        random=0.0,
        ptdr=0.0
    )

    vehicles_path = "../benchmarks/od-matrices/INPUT-od-matrix-10-vehicles.parquet"
    simulator = prepare_simulator(common_args, vehicles_path, alternatives_ratio, route_selection_ratio)
    return simulator


@pytest.fixture
def fastest_alt_provider():
    return FastestPathsAlternatives()


@pytest.fixture
def distributed_alt_provider():
    zmq_ctx = ZeroMqContext()
    port = 5555
    broadcast_port = 5556
    return ZeroMQDistributedAlternatives(
        client=zmq_ctx.get_or_create_client(port=port, broadcast_port=broadcast_port))


def test_compute_alternatives(setup_vehicle, setup_simulator, distributed_alt_provider, fastest_alt_provider):
    vehicles = [setup_vehicle]
    routing_map = setup_simulator.sim.routing_map
    k = 3

    distributed_alt_provider.load_map(routing_map)
    fastest_alt_provider.load_map(routing_map)

    py_alternatives = fastest_alt_provider.compute_alternatives(vehicles, k)
    cpp_alternatives = distributed_alt_provider.compute_alternatives(vehicles, k)

    # one vehicle - 3 alternatives
    assert len(py_alternatives) == 1
    assert len(py_alternatives[0]) == 3
    assert py_alternatives[0][0][0] == expected_alt
    py_travel_time = routing_map.get_path_travel_time(py_alternatives[0][0][0])
    assert round(py_travel_time) == expected_travel_time

    assert len(cpp_alternatives) == 1
    assert len(cpp_alternatives[0]) == 3
    assert cpp_alternatives[0][0][0] == expected_alt

    cpp_travel_time_p = routing_map.get_path_travel_time(cpp_alternatives[0][0][0])
    assert round(cpp_travel_time_p) == expected_travel_time

    cpp_travel_time = cpp_alternatives[0][0][1]
    if cpp_travel_time:
        assert round(cpp_alternatives[0][0][1]) == expected_travel_time


def test_compute_alt_with_map_update(setup_vehicle, setup_simulator, distributed_alt_provider, fastest_alt_provider):
    # do the same as before but with a map update
    vehicles = [setup_vehicle]
    routing_map = setup_simulator.sim.routing_map
    k = 1

    distributed_alt_provider.load_map(routing_map)
    fastest_alt_provider.load_map(routing_map)

    # Change speed on (27349583, 27350859) segment
    segment = routing_map.get_osm_segment(27349583, 27350859)
    new_speed = SpeedKph(segment.max_allowed_speed_kph * 0.6)
    current_travel_time = routing_map.get_path_travel_time([27349583, 27350859])
    expected_travel_time_change = ((segment.length / speed_kph_to_mps(new_speed))
                                   - (segment.length / speed_kph_to_mps(segment.max_allowed_speed_kph)))

    # Alts before map change
    _ = distributed_alt_provider.compute_alternatives(vehicles, k)

    # Update speeds
    updated_speeds = {(27349583, 27350859): new_speed}
    new_speeds = routing_map.update_current_speeds(updated_speeds)
    distributed_alt_provider.update_map(routing_map, new_speeds)

    new_travel_time = routing_map.get_path_travel_time([27349583, 27350859])
    assert isclose(new_travel_time, current_travel_time + expected_travel_time_change, abs_tol=1)

    # Compute alternatives
    cpp_alternatives = distributed_alt_provider.compute_alternatives(vehicles, k)
    py_alternatives = fastest_alt_provider.compute_alternatives(vehicles, k)

    # Check that alt is the same
    assert cpp_alternatives[0][0][0] == expected_alt
    assert py_alternatives[0][0][0] == expected_alt

    cpp_travel_time = routing_map.get_path_travel_time(cpp_alternatives[0][0][0])
    assert isclose(cpp_travel_time, expected_travel_time + expected_travel_time_change, abs_tol=1)
    cpp_travel_time = cpp_alternatives[0][0][1]
    if cpp_travel_time:
        assert isclose(cpp_travel_time, expected_travel_time + expected_travel_time_change, abs_tol=1)
    else:
        print("Travel time not received from evkit")
