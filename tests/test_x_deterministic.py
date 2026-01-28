from time import sleep
import os
import pytest

from datetime import datetime, timedelta

from ruth.simulator import Simulation
from ruth.simulator.kernels import ShortestPathsAlternatives, FirstRouteSelection, FastestPathsAlternatives, \
    RandomRouteSelection
from ruth.tools.simulator import CommonArgs, prepare_simulator, AlternativesRatio, RouteSelectionRatio


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


# set vehicle_path as constant
vehicles_path_10 = "../benchmarks/od-matrices/INPUT-od-matrix-10-vehicles.parquet"


def run_inner_mock(common_args, vehicles_path, alternatives_ratio, route_selection_ratio, name,
                   distributed_alt_provider=None):
    simulator = prepare_simulator(common_args, vehicles_path, alternatives_ratio, route_selection_ratio)
    end_step_fns = []

    alternatives_providers = [ShortestPathsAlternatives(), FastestPathsAlternatives()]
    if distributed_alt_provider is not None:
        alternatives_providers.append(distributed_alt_provider)

    route_selection_providers = [FirstRouteSelection(), RandomRouteSelection()]

    if os.path.exists(f"fcd_history.h5"):
        raise FileExistsError("fcd_history.h5 already exists. Remove it before running the test.")


    simulator.simulate(
        alternatives_providers=alternatives_providers,
        route_selection_providers=route_selection_providers,
        end_step_fns=end_step_fns,
    )

    os.rename("fcd_history.h5", f"fcd_history_{name}.h5")

    return simulator


def compare_simulation(simulation1, simulation2, sim_1_name="sim1", sim_2_name="sim2"):
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
    fcd_history_1 = Simulation.load_h5_df(f"fcd_history_{sim_1_name}.h5")["df"]
    fcd_history_2 = Simulation.load_h5_df(f"fcd_history_{sim_2_name}.h5")["df"]
    assert len(fcd_history_1) == len(fcd_history_2), "FCD history mismatch"
    for i in range(len(fcd_history_1)):
        assert fcd_history_1.iloc[i]['timestamp'] == fcd_history_2.iloc[i]['timestamp'], f"Datetime mismatch at index {i}"
        assert fcd_history_1.iloc[i]['node_from'] == fcd_history_2.iloc[i]['node_from'], f" Node from mismatch at index {i}"
        assert fcd_history_1.iloc[i]['node_to'] == fcd_history_2.iloc[i]['node_to'], f"Node to mismatch at index {i}"
        assert fcd_history_1.iloc[i]['speed_mps'] == fcd_history_2.iloc[i]['speed_mps'], f"Speed mismatch at index {i}"
        assert fcd_history_1.iloc[i]['vehicle_id'] == fcd_history_2.iloc[i]['vehicle_id'], f"Vehicle ID mismatch at index {i}"
        assert fcd_history_1.iloc[i]['start_offset_m'] == fcd_history_2.iloc[i]['start_offset_m'], f"Start offset mismatch at index {i}"
        assert fcd_history_1.iloc[i]['active'] == fcd_history_2.iloc[i]['active'], f"Active mismatch at index {i}"


def test_simulation(setup_common_args):
    setup_alternatives_ratio = AlternativesRatio(
        default=1.0,
        dijkstra_fastest=0.0,
        dijkstra_shortest=0.0,
        plateau_fastest=0.0
    )
    setup_route_selection_ratio = RouteSelectionRatio(no_alternative=1.0, first=0.0, random=0.0, ptdr=0.0)

    simulator1 = run_inner_mock(setup_common_args, vehicles_path_10, setup_alternatives_ratio,
                                setup_route_selection_ratio, "sim1")

    assert simulator1.current_offset is None
    simulation1 = simulator1.state

    # check if all vehicles finished
    for vehicle in simulation1.vehicles:
        assert vehicle.active is False
        assert vehicle.start_index == len(vehicle.osm_route) - 2  # last segment

    simulator2 = run_inner_mock(setup_common_args, vehicles_path_10, setup_alternatives_ratio,
                                setup_route_selection_ratio, "sim2")

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
                                setup_route_selection_ratio, "sim1")

    assert simulator1.current_offset is None
    simulation1 = simulator1.state

    map_path = simulator1.sim.routing_map.file_path
    os.remove(map_path)

    simulator2 = run_inner_mock(setup_common_args, vehicles_path_10, setup_alternatives_ratio,
                                setup_route_selection_ratio, "sim2")

    simulation2 = simulator2.state
    compare_simulation(simulation1, simulation2)


def test_simulation_with_alt(setup_common_args, setup_alternatives_ratio, setup_route_selection_ratio):
    simulator1 = run_inner_mock(setup_common_args, vehicles_path_10, setup_alternatives_ratio,
                                setup_route_selection_ratio, "sim1")

    assert simulator1.current_offset is None
    simulation1 = simulator1.state

    # check if all vehicles finished
    for vehicle in simulation1.vehicles:
        assert vehicle.active is False
        assert vehicle.start_index == len(vehicle.osm_route) - 2  # last segment

    simulator2 = run_inner_mock(setup_common_args, vehicles_path_10, setup_alternatives_ratio,
                                setup_route_selection_ratio, "sim2")

    simulation2 = simulator2.state
    compare_simulation(simulation1, simulation2)