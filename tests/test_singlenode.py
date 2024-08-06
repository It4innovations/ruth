from unittest.mock import patch, MagicMock

import pytest

from datetime import datetime, timedelta

from ruth.data.segment import LengthMeters
from ruth.simulator.kernels import ZeroMQDistributedAlternatives, ShortestPathsAlternatives
from ruth.tools.simulator import CommonArgs, prepare_simulator, AlternativesRatio, RouteSelectionRatio, ZeroMqContext

from ruth.vehicle import Vehicle


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


def test_vehicle_within_offset(setup_simulator):
    sim = setup_simulator.state

    sim_offset = setup_simulator.current_offset
    sim_offset_seconds = sim_offset.total_seconds()
    frequency = sim.setting.round_freq
    frequency_seconds = frequency.total_seconds()

    # find max value that rounds to this offset
    max_offset = sim_offset + frequency // 2
    max_offset_seconds = max_offset.total_seconds()

    all_vehicles_offset = [v.time_offset.total_seconds() for v in sim.vehicles]
    all_vehicles_offset = sorted(all_vehicles_offset)
    all_offsets_rounded = [round(offset / frequency_seconds) * frequency_seconds for offset in all_vehicles_offset]

    to_be_moved_rounded = [offset for offset in all_offsets_rounded if offset == sim_offset_seconds]
    to_be_moved_by_max = [offset for offset in all_offsets_rounded if offset < max_offset_seconds]

    expected_count = len(to_be_moved_rounded)

    vehicles_to_be_moved = [v for v in sim.vehicles if sim.is_vehicle_within_offset(v, sim_offset)]
    assert len(vehicles_to_be_moved) == expected_count
    assert len(to_be_moved_by_max) == expected_count


@patch('logging.Logger.error')
def test_stuck_detection(mock, setup_vehicle, setup_simulator):
    # test stuck detection
    sim = setup_simulator.state

    # let run for 3x round_freq
    setup_simulator.sim.setting.stuck_detection = 3
    stuck_limit = sim.setting.round_freq * sim.setting.stuck_detection
    last_time_moved = sim.compute_current_offset()
    stop_offset = last_time_moved + stuck_limit
    print(stop_offset)
    print(stop_offset - last_time_moved)

    # Set up vehicle at the end of the segment wanting to move to the next segment
    node_from, node_to = setup_vehicle.osm_route[0:2]
    segment_length = sim.routing_map.segment_lengths[(node_from, node_to)]
    setup_vehicle.start_distance_offset = segment_length
    sim.queues_manager.add_to_queue(setup_vehicle)
    sim.vehicles = [setup_vehicle]

    # next segment is blocked
    next_node_from, next_node_to = setup_vehicle.osm_route[1:3]
    sim.routing_map.current_network[next_node_from][next_node_to]['speed_kph'] = 0.0

    # TEST stuck detection
    setup_simulator.simulate([], [])
    mock.assert_called_with("The simulation is stuck at 0:02:15.")

    assert setup_vehicle.start_index == 0
    assert setup_vehicle.start_distance_offset == segment_length


# TODO: test update map speeds - temporary max and current
def test_update_map_speeds_no_overlap(setup_simulator):
    sim = setup_simulator.state
    last_map_update = setup_simulator.current_offset - timedelta(seconds=1)
    sim.setting.map_update_freq_s = timedelta(seconds=5)

    alt_provider = [MagicMock(ShortestPathsAlternatives)]

    # Time 0:00
    temp_speeds = {(25664661, 27349583): 0.0}
    updated_speeds = {}
    with patch.object(sim.routing_map, 'update_temporary_max_speeds', return_value=temp_speeds):
        updated_speeds, last_map_update = setup_simulator.update_map_speeds(updated_speeds, last_map_update, alt_provider)

    assert updated_speeds == temp_speeds
    assert last_map_update == (setup_simulator.current_offset - timedelta(seconds=1))

    # Time 0:05
    gv_speeds = {(27349583, 27350859): 30.0}
    setup_simulator.current_offset = setup_simulator.current_offset + sim.setting.map_update_freq_s
    with patch.object(sim.routing_map, 'update_temporary_max_speeds', return_value={}):
        with patch.object(sim.global_view, 'take_segment_speeds', return_value=gv_speeds):
            updated_speeds, last_map_update = setup_simulator.update_map_speeds(updated_speeds, last_map_update, alt_provider)

    both_speeds = {**temp_speeds, **gv_speeds}
    alt_provider[0].update_map.assert_called_once_with(sim.routing_map, both_speeds)
    assert last_map_update == setup_simulator.current_offset
    assert updated_speeds == {}


# TODO: test sim update fcds
# TODO: test drop old records
# TODO: test change baseline alts

# TODO: QUEUES in new file
# TODO: VEHICLE file
# TDOD: GLOBAL VIEW file
# TODO: MAP file
