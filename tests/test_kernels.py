from cmath import isclose
import os

import pytest

from datetime import datetime, timedelta, timezone

from ruth.data.segment import LengthMeters, SpeedKph, speed_kph_to_mps
from ruth.simulator.common import load_vehicles
from ruth.simulator.kernels import FastestPathsAlternatives
# from ruth.simulator.kernels import ZeroMQDistributedAlternatives
from ruth.tools.simulator import CommonArgs, prepare_simulator, AlternativesRatio, RouteSelectionRatio, ZeroMqContext

vehicles_path = os.path.join(
    os.path.dirname(__file__),
    "../benchmarks/od-matrices/INPUT-od-matrix-10-vehicles.parquet",
)


@pytest.fixture
def setup_vehicle():
    vehicles, _, _ = load_vehicles(vehicles_path)
    return vehicles[0]


@pytest.fixture
def setup_simulator():
    common_args = CommonArgs(
        task_id="test",
        departure_time=datetime(2021, 1, 1, 0, 0, 0),
        round_frequency=timedelta(seconds=5),
        k_alternatives=1,
        map_update_freq=timedelta(seconds=1),
        los_vehicles_tolerance=timedelta(seconds=0),
        travel_time_limit_perc=1.5,
        speeds_path=None,
        out="./output",
        seed=42,
        walltime=None,
        saving_interval=None,
        continue_from="",
        stuck_detection=5,
        plateau_default_route=False,
        buffer_size=1000,
        max_records_per_file=10000,
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

    simulator = prepare_simulator(common_args, vehicles_path, alternatives_ratio, route_selection_ratio)
    return simulator


@pytest.fixture
def fastest_alt_provider():
    return FastestPathsAlternatives()


# @pytest.fixture
# def distributed_alt_provider():
#     zmq_ctx = ZeroMqContext()
#     port = 5555
#     broadcast_port = 5556
#     return ZeroMQDistributedAlternatives(
#         client=zmq_ctx.get_or_create_client(port=port, broadcast_port=broadcast_port))


def test_compute_alternatives(setup_vehicle, setup_simulator, fastest_alt_provider):
    vehicles = [setup_vehicle]
    routing_map = setup_simulator.sim.routing_map
    k = 3

    # distributed_alt_provider.load_map(routing_map)
    fastest_alt_provider.load_map(routing_map)

    py_alternatives = fastest_alt_provider.compute_alternatives(vehicles, k)
    # cpp_alternatives = distributed_alt_provider.compute_alternatives(vehicles, k)

    # one vehicle - 3 alternatives
    assert len(py_alternatives) == 1
    assert len(py_alternatives[0]) == 3
    assert py_alternatives[0][0][0] == setup_vehicle.osm_route
    py_travel_time = routing_map.get_path_travel_time(py_alternatives[0][0][0])
    expected_travel_time = routing_map.get_path_travel_time(setup_vehicle.osm_route)
    assert round(py_travel_time) == round(expected_travel_time)

    # assert len(cpp_alternatives) == 1
    # assert len(cpp_alternatives[0]) == 3
    # assert cpp_alternatives[0][0][0] == expected_alt
    #
    # cpp_travel_time_p = routing_map.get_path_travel_time(cpp_alternatives[0][0][0])
    # assert round(cpp_travel_time_p) == expected_travel_time
    #
    # cpp_travel_time = cpp_alternatives[0][0][1]
    # if cpp_travel_time:
    #     assert round(cpp_alternatives[0][0][1]) == expected_travel_time


def test_compute_alt_with_map_update(setup_vehicle, setup_simulator, fastest_alt_provider):
    # do the same as before but with a map update
    vehicles = [setup_vehicle]
    routing_map = setup_simulator.sim.routing_map
    k = 1

    # distributed_alt_provider.load_map(routing_map)
    fastest_alt_provider.load_map(routing_map)

    node_from, node_to = setup_vehicle.osm_route[:2]
    segment = routing_map.get_osm_segment(node_from, node_to)
    new_speed = SpeedKph(segment.max_allowed_speed_kph * 0.6)
    current_travel_time = routing_map.get_path_travel_time([node_from, node_to])
    expected_travel_time_change = ((segment.length / speed_kph_to_mps(new_speed))
                                   - (segment.length / speed_kph_to_mps(segment.max_allowed_speed_kph)))

    # Alts before map change
    # _ = distributed_alt_provider.compute_alternatives(vehicles, k)

    # Update speeds
    updated_speeds = {(node_from, node_to): new_speed}
    new_speeds = routing_map.update_current_speeds(updated_speeds)
    # distributed_alt_provider.update_map(new_speeds)

    new_travel_time = routing_map.get_path_travel_time([node_from, node_to])
    assert isclose(new_travel_time, current_travel_time + expected_travel_time_change, abs_tol=1)

    # Compute alternatives
    # cpp_alternatives = distributed_alt_provider.compute_alternatives(vehicles, k)
    py_alternatives = fastest_alt_provider.compute_alternatives(vehicles, k)

    updated_route = py_alternatives[0][0][0]
    assert updated_route[0] == setup_vehicle.origin_node
    assert updated_route[-1] == setup_vehicle.dest_node
    assert (node_from, node_to) not in zip(updated_route[:-1], updated_route[1:])

    # cpp_travel_time = routing_map.get_path_travel_time(cpp_alternatives[0][0][0])
    # assert isclose(cpp_travel_time, expected_travel_time + expected_travel_time_change, abs_tol=1)
    # cpp_travel_time = cpp_alternatives[0][0][1]
    # if cpp_travel_time:
    #     assert isclose(cpp_travel_time, expected_travel_time + expected_travel_time_change, abs_tol=1)
    # else:
    #     print("Travel time not received from evkit")
