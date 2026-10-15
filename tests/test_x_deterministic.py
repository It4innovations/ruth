import glob
import json
import os
import pytest
import pandas as pd

from datetime import datetime, timedelta
from pathlib import Path

from ruth.simulator import Simulation
from ruth.simulator.kernels import ShortestPathsAlternatives, FirstRouteSelection, FastestPathsAlternatives, \
    RandomRouteSelection, MPIDistributedAlternatives
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
        stuck_detection=5,
        travel_time_limit_perc=1.5,
        speeds_path=None,
        out="./output",
        seed=42,
        walltime=None,
        saving_interval=None,
        continue_from="",
        plateau_default_route=False,
        buffer_size=1000,
        max_records_per_file=10000,
        map_graphml=str(Path(__file__).resolve().parent / "data" /
                        "50_16568920000002-14_321441000000016-50_020240399999985-"
                        "14_592499399999983_2024-01-10T00-00-00.graphml"),
    )


@pytest.fixture(params=[[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
def setup_alternatives_ratio(request):
    params = request.param
    if params[3] > 0:
        request.getfixturevalue("native_scheduler")
    return AlternativesRatio(
        default=params[0],
        dijkstra_fastest=params[1],
        dijkstra_shortest=params[2],
        plateau_fastest=params[3]
    )


@pytest.fixture(scope="session")
def native_scheduler():
    from ruth.mpi_comm.distributor import MPIDistributor
    with MPIDistributor(threads=2):
        yield


@pytest.fixture
def setup_route_selection_ratio():
    return RouteSelectionRatio(
        no_alternative=0.0,
        first=1.0,
        random=0.0,
        ptdr=0.0
    )


test_graphml_path = (
    Path(__file__).resolve().parent / "data"
    / "50_16568920000002-14_321441000000016-50_020240399999985-"
      "14_592499399999983_2024-01-10T00-00-00.graphml"
)


def write_bucket_dataset_copy(single_path, dataset_path, partition_seconds=10):
    df = pd.read_parquet(single_path, engine="fastparquet")
    df["time_offset_s"] = df["time_offset"].dt.total_seconds().astype("int64")
    df["start_bucket_s"] = (df["time_offset_s"] // partition_seconds) * partition_seconds

    first_row = df.iloc[0]
    manifest = {
        "shared_columns": {
            "frequency": int(first_row["frequency"].total_seconds()),
            "fcd_sampling_period": int(first_row["fcd_sampling_period"].total_seconds()),
            "download_date": first_row["download_date"],
            "bbox": {
                "lat_max": first_row["bbox_lat_max"],
                "lon_min": first_row["bbox_lon_min"],
                "lat_min": first_row["bbox_lat_min"],
                "lon_max": first_row["bbox_lon_max"],
            },
        }
    }

    dataset_path.mkdir()
    (dataset_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for bucket, bucket_df in df.groupby("start_bucket_s", sort=True):
        bucket_path = dataset_path / f"start_bucket_s={int(bucket)}"
        bucket_path.mkdir()
        bucket_df = bucket_df.drop(columns=["frequency", "fcd_sampling_period"])
        bucket_df.to_parquet(bucket_path / "part.0.parquet", engine="fastparquet", index=False)


def prepare_input_equivalence_run_dir(run_dir):
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True)
    (data_dir / test_graphml_path.name).symlink_to(test_graphml_path)


def run_input_equivalence_simulation(common_args, vehicles_path, alternatives_ratio,
                                     route_selection_ratio, run_dir, name):
    prepare_input_equivalence_run_dir(run_dir)

    cwd = Path.cwd()
    os.chdir(run_dir)
    try:
        simulator = run_inner_mock(common_args, str(vehicles_path), alternatives_ratio,
                                   route_selection_ratio, name)
        step_summary = [
            (step.simulation_offset, step.step, step.n_active, step.need_new_route)
            for step in simulator.state.steps_info
        ]
        return step_summary, load_fcd_history(name)
    finally:
        os.chdir(cwd)


# check history - handle both single file and partitioned files
def load_fcd_history(name):
    import glob
    import pandas as pd
    # Try loading single file first
    if os.path.exists(f"fcd_history_{name}.h5"):
        return Simulation.load_h5_df(f"fcd_history_{name}.h5")["df"]

    # Try loading partitioned files
    part_files = sorted(glob.glob(f"fcd_history_{name}-part*.h5"))
    if part_files:
        dfs = []
        for part_file in part_files:
            df = Simulation.load_h5_df(part_file)["df"]
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    raise FileNotFoundError(f"No fcd_history files found for {name}")

def run_inner_mock(common_args, vehicles_path, alternatives_ratio, route_selection_ratio, name,
                   distributed_alt_provider=None, end_step_fns=None):
    simulator = prepare_simulator(common_args, vehicles_path, alternatives_ratio, route_selection_ratio)
    end_step_fns = end_step_fns or []

    alternatives_providers = [ShortestPathsAlternatives(), FastestPathsAlternatives()]
    if alternatives_ratio.plateau_fastest > 0:
        alternatives_providers.append(MPIDistributedAlternatives())
    if distributed_alt_provider is not None:
        alternatives_providers.append(distributed_alt_provider)

    route_selection_providers = [FirstRouteSelection(), RandomRouteSelection()]

    if os.path.exists(f"fcd_history.h5"):
        raise FileExistsError("fcd_history.h5 already exists. Remove it before running the test.")

    existing_parts = glob.glob("fcd_history-part*.h5")
    if existing_parts:
        raise FileExistsError(f"fcd_history part files already exist. Remove them before running the test.")

    simulator.simulate(
        alternatives_providers=alternatives_providers,
        route_selection_providers=route_selection_providers,
        end_step_fns=end_step_fns,
    )

    if os.path.exists("fcd_history.h5"):
        os.rename("fcd_history.h5", f"fcd_history_{name}.h5")
    else:
        part_files = sorted(glob.glob("fcd_history-part*.h5"))
        if part_files:
            # Rename all part files
            for i, part_file in enumerate(part_files):
                new_name = f"fcd_history_{name}-part{i:04d}.h5"
                os.rename(part_file, new_name)
        else:
            raise FileNotFoundError("No fcd_history files were created during simulation")

    return simulator


def compare_simulation(simulation1, simulation2, sim_1_name="sim1", sim_2_name="sim2",
                       fcd_history_1=None, fcd_history_2=None):
    assert len(simulation1.steps_info) == len(simulation2.steps_info), "Steps info mismatch"
    assert simulation1.queues_manager.queues == simulation2.queues_manager.queues, "Queues mismatch"

    assert [v.id for v in simulation1.vehicles] == [v.id for v in simulation2.vehicles]

    # check vehicles
    for i in range(len(simulation1.vehicles)):
        assert simulation1.vehicles[i].osm_route == simulation2.vehicles[i].osm_route, f"Route mismatch at index {i}"
        assert simulation1.vehicles[i].time_offset == simulation2.vehicles[
            i].time_offset, f"Time offset mismatch at index {i}"

        assert simulation1.vehicles[i].start_index == simulation2.vehicles[i].start_index
        assert simulation1.vehicles[i].start_distance_offset == simulation2.vehicles[i].start_distance_offset
        assert simulation1.vehicles[i].active == simulation2.vehicles[i].active
        assert simulation1.vehicles[i].status == simulation2.vehicles[i].status

    # check steps_info
    for i in range(len(simulation1.steps_info)):
        assert simulation1.steps_info[i].simulation_offset == simulation2.steps_info[
            i].simulation_offset, f"Offset mismatch at index {i}"
        assert simulation1.steps_info[i].step == simulation2.steps_info[i].step, f"Step mismatch at index {i}"
        assert simulation1.steps_info[i].n_active == simulation2.steps_info[i].n_active, f"Active mismatch at index {i}"
        assert simulation1.steps_info[i].need_new_route == simulation2.steps_info[
            i].need_new_route, f"Need new route mismatch at index {i}"

    if fcd_history_1 is None:
        fcd_history_1 = load_fcd_history(sim_1_name)
    if fcd_history_2 is None:
        fcd_history_2 = load_fcd_history(sim_2_name)
    pd.testing.assert_frame_equal(fcd_history_1.reset_index(drop=True),
                                  fcd_history_2.reset_index(drop=True), check_exact=True)


def test_simulation(setup_common_args, regression_vehicles_path):
    setup_alternatives_ratio = AlternativesRatio(
        default=1.0,
        dijkstra_fastest=0.0,
        dijkstra_shortest=0.0,
        plateau_fastest=0.0
    )
    setup_route_selection_ratio = RouteSelectionRatio(no_alternative=1.0, first=0.0, random=0.0, ptdr=0.0)

    simulator1 = run_inner_mock(setup_common_args, regression_vehicles_path, setup_alternatives_ratio,
                                setup_route_selection_ratio, "sim1")

    assert simulator1.current_offset is None
    simulation1 = simulator1.state

    # check if all vehicles finished
    for vehicle in simulation1.vehicles:
        assert vehicle.active is False
        assert vehicle.start_index == len(vehicle.osm_route) - 2  # last segment

    simulator2 = run_inner_mock(setup_common_args, regression_vehicles_path, setup_alternatives_ratio,
                                setup_route_selection_ratio, "sim2")

    simulation2 = simulator2.state
    compare_simulation(simulation1, simulation2)


def test_simulation_with_replaced_map(setup_common_args, tmp_path, regression_vehicles_path):
    setup_alternatives_ratio = AlternativesRatio(
        default=1.0,
        dijkstra_fastest=0.0,
        dijkstra_shortest=0.0,
        plateau_fastest=0.0
    )
    setup_route_selection_ratio = RouteSelectionRatio(no_alternative=1.0, first=0.0, random=0.0, ptdr=0.0)

    # Work with a private copy: this test must never remove the shared map fixture.
    import shutil
    private_map = tmp_path / test_graphml_path.name
    shutil.copyfile(test_graphml_path, private_map)
    setup_common_args.map_graphml = str(private_map)
    simulator1 = run_inner_mock(setup_common_args, regression_vehicles_path, setup_alternatives_ratio,
                                setup_route_selection_ratio, "sim1")

    assert simulator1.current_offset is None
    simulation1 = simulator1.state

    os.remove(private_map)
    shutil.copyfile(test_graphml_path, private_map)

    simulator2 = run_inner_mock(setup_common_args, regression_vehicles_path, setup_alternatives_ratio,
                                setup_route_selection_ratio, "sim2")

    simulation2 = simulator2.state
    compare_simulation(simulation1, simulation2)


def test_simulation_with_alt(setup_common_args, setup_alternatives_ratio, setup_route_selection_ratio,
                             regression_vehicles_path):
    simulator1 = run_inner_mock(setup_common_args, regression_vehicles_path, setup_alternatives_ratio,
                                setup_route_selection_ratio, "sim1")

    assert simulator1.current_offset is None
    simulation1 = simulator1.state

    # check if all vehicles finished
    for vehicle in simulation1.vehicles:
        assert vehicle.active is False
        assert vehicle.start_index == len(vehicle.osm_route) - 2  # last segment

    simulator2 = run_inner_mock(setup_common_args, regression_vehicles_path, setup_alternatives_ratio,
                                setup_route_selection_ratio, "sim2")

    simulation2 = simulator2.state
    compare_simulation(simulation1, simulation2)


def test_single_file_and_bucket_simulations_emit_same_fcd_history(setup_common_args, tmp_path,
                                                               regression_vehicles_path):
    dataset_path = tmp_path / "vehicles_dataset"
    write_bucket_dataset_copy(Path(regression_vehicles_path), dataset_path)

    alternatives_ratio = AlternativesRatio(
        default=0.0,
        dijkstra_fastest=0.5,
        dijkstra_shortest=0.5,
        plateau_fastest=0.0,
    )
    route_selection_ratio = RouteSelectionRatio(
        no_alternative=0.0,
        first=1.0,
        random=0.0,
        ptdr=0.0,
    )

    single_steps, single_fcd = run_input_equivalence_simulation(
        setup_common_args,
        Path(regression_vehicles_path),
        alternatives_ratio,
        route_selection_ratio,
        tmp_path / "single_run",
        "single",
    )
    dataset_steps, dataset_fcd = run_input_equivalence_simulation(
        setup_common_args,
        dataset_path,
        alternatives_ratio,
        route_selection_ratio,
        tmp_path / "dataset_run",
        "dataset",
    )

    assert single_steps == dataset_steps
    pd.testing.assert_frame_equal(
        single_fcd.reset_index(drop=True),
        dataset_fcd.reset_index(drop=True),
    )


def test_continue_from_matches_uninterrupted(setup_common_args, setup_alternatives_ratio,
                                            setup_route_selection_ratio, regression_vehicles_path,
                                            tmp_path):
    full = run_inner_mock(setup_common_args, regression_vehicles_path, setup_alternatives_ratio,
                          setup_route_selection_ratio, "full")
    partial = run_inner_mock(setup_common_args, regression_vehicles_path, setup_alternatives_ratio,
                             setup_route_selection_ratio, "partial",
                             end_step_fns=[lambda state: state.number_of_steps >= 5])
    assert not partial.state.finished()
    checkpoint = tmp_path / "resume.pickle"
    partial.state.store(checkpoint)
    setup_common_args.continue_from = str(checkpoint)
    resumed = run_inner_mock(setup_common_args, regression_vehicles_path, setup_alternatives_ratio,
                             setup_route_selection_ratio, "resumed")
    assert resumed.state.finished()
    combined = pd.concat([load_fcd_history("partial"), load_fcd_history("resumed")], ignore_index=True)
    compare_simulation(full.state, resumed.state, fcd_history_1=load_fcd_history("full"),
                       fcd_history_2=combined)


def test_continue_from_accepts_existing_fcd_parts(setup_common_args, regression_vehicles_path, tmp_path):
    alternatives = AlternativesRatio(default=1.0, dijkstra_fastest=0.0,
                                     dijkstra_shortest=0.0, plateau_fastest=0.0)
    selection = RouteSelectionRatio(no_alternative=1.0, first=0.0, random=0.0, ptdr=0.0)
    simulator = prepare_simulator(setup_common_args, regression_vehicles_path, alternatives, selection)
    simulator.simulate([], [], max_steps=2)
    checkpoint = tmp_path / "existing-history.pickle"
    simulator.state.store(checkpoint)
    existing = list(Path.cwd().glob("fcd_history-part*.h5"))
    assert existing
    setup_common_args.continue_from = str(checkpoint)
    resumed = prepare_simulator(setup_common_args, regression_vehicles_path, alternatives, selection)
    assert resumed.state.number_of_steps == 2
    assert all(path.exists() for path in existing)


def test_checkpoint_does_not_duplicate_flushed_history(setup_common_args, regression_vehicles_path, tmp_path):
    from ruth.simulator import SingleNodeSimulator
    alternatives = AlternativesRatio(default=1.0, dijkstra_fastest=0.0,
                                     dijkstra_shortest=0.0, plateau_fastest=0.0)
    selection = RouteSelectionRatio(no_alternative=1.0, first=0.0, random=0.0, ptdr=0.0)
    simulator = prepare_simulator(setup_common_args, regression_vehicles_path, alternatives, selection)
    checkpoint = tmp_path / "buffered-history.pickle"

    def stop_and_save(state):
        if state.number_of_steps == 2:
            state.store(checkpoint)
            return True
        return False

    simulator.simulate([], [], end_step_fns=[stop_and_save])
    # Load directly to isolate serialization from the separate CLI file-existence bug.
    resumed = SingleNodeSimulator(Simulation.load(checkpoint))
    resumed.simulate([], [], max_steps=3)
    files = sorted(Path.cwd().glob("fcd_history-part*.h5"))
    combined = pd.concat([Simulation.load_h5_df(str(path))["df"] for path in files], ignore_index=True)
    assert not combined.duplicated().any(), "Checkpoint replay duplicated FCD rows already flushed by graceful stop"
