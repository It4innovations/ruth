from time import sleep
import os
import pytest

from datetime import datetime, timedelta

from ruth.simulator.kernels import ShortestPathsAlternatives, FirstRouteSelection, FastestPathsAlternatives, \
    RandomRouteSelection, ZeroMQDistributedAlternatives
from ruth.tools.simulator import CommonArgs, prepare_simulator, AlternativesRatio, RouteSelectionRatio, ZeroMqContext


@pytest.fixture
def setup_common_args():
    return CommonArgs(
        task_id="test",
        departure_time=datetime(2021, 1, 1, 0, 0, 0),
        round_frequency=timedelta(seconds=5),
        k_alternatives=1,
        map_update_freq=timedelta(hours=10),
        los_vehicles_tolerance=timedelta(seconds=5),
        stuck_detection=5
    )


@pytest.fixture(params=[[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
def setup_alternatives_ratio(request):
    params = request.param
    return AlternativesRatio(
        default=params[0],
        dijkstra_fastest=params[1],
        dijkstra_shortest=params[2],
        plateau_fastest=params[3]
    )


@pytest.fixture
def setup_route_selection_ratio():
    return RouteSelectionRatio(
        no_alternative=0.0,
        first=1.0,
        random=0.0,
        ptdr=0.0
    )


@pytest.fixture
def setup_distributed_alt_provider():
    zmq_ctx = ZeroMqContext()
    port = 5555
    broadcast_port = 5556
    return ZeroMQDistributedAlternatives(
        client=zmq_ctx.get_or_create_client(port=port, broadcast_port=broadcast_port))


# set vehicle_path as constant
vehicles_path_10 = "../benchmarks/od-matrices/INPUT-od-matrix-10-vehicles.parquet"


def run_inner_mock(common_args, vehicles_path, alternatives_ratio, route_selection_ratio,
                   distributed_alt_provider=None):
    simulator = prepare_simulator(common_args, vehicles_path, alternatives_ratio, route_selection_ratio)
    end_step_fns = []

    alternatives_providers = [ShortestPathsAlternatives(), FastestPathsAlternatives()]
    if distributed_alt_provider is not None:
        alternatives_providers.append(distributed_alt_provider)

    route_selection_providers = [FirstRouteSelection(), RandomRouteSelection()]

    simulator.simulate(
        alternatives_providers=alternatives_providers,
        route_selection_providers=route_selection_providers,
        end_step_fns=end_step_fns,
    )

    return simulator


def compare_simulation(simulation1, simulation2):
    assert len(simulation1.history.fcd_history) == len(simulation2.history.fcd_history), "FCD history mismatch"
    assert len(simulation1.steps_info) == len(simulation2.steps_info), "Steps info mismatch"
    assert simulation1.queues_manager.queues == simulation2.queues_manager.queues, "Queues mismatch"

    # check vehicles
    for i in range(len(simulation1.vehicles)):
        assert simulation1.vehicles[i].osm_route == simulation2.vehicles[i].osm_route, f"Route mismatch at index {i}"
        assert simulation1.vehicles[i].time_offset == simulation2.vehicles[
            i].time_offset, f"Time offset mismatch at index {i}"

    # check steps_info
    for i in range(len(simulation1.steps_info)):
        assert simulation1.steps_info[i].simulation_offset == simulation2.steps_info[
            i].simulation_offset, f"Offset mismatch at index {i}"
        assert simulation1.steps_info[i].step == simulation2.steps_info[i].step, f"Step mismatch at index {i}"
        assert simulation1.steps_info[i].n_active == simulation2.steps_info[i].n_active, f"Active mismatch at index {i}"
        assert simulation1.steps_info[i].need_new_route == simulation2.steps_info[
            i].need_new_route, f"Need new route mismatch at index {i}"

    # check history
    for i in range(len(simulation1.history.fcd_history)):
        assert simulation1.history.fcd_history[i].datetime == simulation2.history.fcd_history[
            i].datetime, f"Datetime mismatch at index {i}"
        assert simulation1.history.fcd_history[i].segment == simulation2.history.fcd_history[
            i].segment, f"Segment mismatch at index {i}"
        assert simulation1.history.fcd_history[i].speed == simulation2.history.fcd_history[
            i].speed, f"Speed mismatch at index {i}"
        assert simulation1.history.fcd_history[i].vehicle_id == simulation2.history.fcd_history[
            i].vehicle_id, f"Vehicle ID mismatch at index {i}"
        assert simulation1.history.fcd_history[i].start_offset == simulation2.history.fcd_history[
            i].start_offset, f"Start offset mismatch at index {i}"
        assert simulation1.history.fcd_history[i].status == simulation2.history.fcd_history[
            i].status, f"Status mismatch at index {i}"
        assert simulation1.history.fcd_history[i].active == simulation2.history.fcd_history[
            i].active, f"Active mismatch at index {i}"


def test_simulation(setup_common_args):
    setup_alternatives_ratio = AlternativesRatio(
        default=1.0,
        dijkstra_fastest=0.0,
        dijkstra_shortest=0.0,
        plateau_fastest=0.0
    )
    setup_route_selection_ratio = RouteSelectionRatio(no_alternative=1.0, first=0.0, random=0.0, ptdr=0.0)

    simulator1 = run_inner_mock(setup_common_args, vehicles_path_10, setup_alternatives_ratio,
                                setup_route_selection_ratio)

    assert simulator1.current_offset is None
    simulation1 = simulator1.state

    # check if all vehicles finished
    for vehicle in simulation1.vehicles:
        assert vehicle.active is False
        assert vehicle.start_index == len(vehicle.osm_route) - 2  # last segment

    simulator2 = run_inner_mock(setup_common_args, vehicles_path_10, setup_alternatives_ratio,
                                setup_route_selection_ratio)

    simulation2 = simulator2.state
    compare_simulation(simulation1, simulation2)


def test_simulation_download_map(setup_common_args):
    setup_alternatives_ratio = AlternativesRatio(
        default=1.0,
        dijkstra_fastest=0.0,
        dijkstra_shortest=0.0,
        plateau_fastest=0.0
    )
    setup_route_selection_ratio = RouteSelectionRatio(no_alternative=1.0, first=0.0, random=0.0, ptdr=0.0)

    simulator1 = run_inner_mock(setup_common_args, vehicles_path_10, setup_alternatives_ratio,
                                setup_route_selection_ratio)

    assert simulator1.current_offset is None
    simulation1 = simulator1.state

    map_path = simulator1.sim.routing_map.file_path
    os.remove(map_path)

    simulator2 = run_inner_mock(setup_common_args, vehicles_path_10, setup_alternatives_ratio,
                                setup_route_selection_ratio)

    simulation2 = simulator2.state
    compare_simulation(simulation1, simulation2)


def test_simulation_with_alt(setup_common_args, setup_alternatives_ratio, setup_route_selection_ratio):
    simulator1 = run_inner_mock(setup_common_args, vehicles_path_10, setup_alternatives_ratio,
                                setup_route_selection_ratio)

    assert simulator1.current_offset is None
    simulation1 = simulator1.state

    # check if all vehicles finished
    for vehicle in simulation1.vehicles:
        assert vehicle.active is False
        assert vehicle.start_index == len(vehicle.osm_route) - 2  # last segment

    simulator2 = run_inner_mock(setup_common_args, vehicles_path_10, setup_alternatives_ratio,
                                setup_route_selection_ratio)

    simulation2 = simulator2.state
    compare_simulation(simulation1, simulation2)


def test_simulation_with_alt_distributed(setup_common_args, setup_route_selection_ratio,
                                         setup_distributed_alt_provider):
    setup_alternatives_ratio = AlternativesRatio(
        default=0.0,
        dijkstra_fastest=0.0,
        dijkstra_shortest=0.0,
        plateau_fastest=1.0
    )
    print("\nRunning first simulation")
    simulator1 = run_inner_mock(setup_common_args, vehicles_path_10, setup_alternatives_ratio,
                                setup_route_selection_ratio, setup_distributed_alt_provider)

    assert simulator1.current_offset is None
    simulation1 = simulator1.state

    # check if all vehicles finished
    for vehicle in simulation1.vehicles:
        assert vehicle.active is False
        assert vehicle.start_index == len(vehicle.osm_route) - 2  # last segment

    print("\nRunning second simulation")
    simulator2 = run_inner_mock(setup_common_args, vehicles_path_10, setup_alternatives_ratio,
                                setup_route_selection_ratio, setup_distributed_alt_provider)

    simulation2 = simulator2.state
    compare_simulation(simulation1, simulation2)
