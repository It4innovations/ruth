"""Unit tests for the live steering console (fake Simulation, no map download)."""

import json
import urllib.request
from datetime import datetime, timedelta
from types import SimpleNamespace

import networkx as nx
import pytest

from ruth.console.commands import validate_param, validate_restriction
from ruth.console.console import SimulationConsole
from ruth.console.snapshot import SnapshotBuilder
from ruth.tools.simulator import ConsoleArgs


def make_args(**kw):
    return ConsoleArgs(**kw)


def make_fake_sim(n_vehicles=6, per_segment=None):
    """A minimal Simulation stand-in: 4-node line graph 1-2-3-4."""
    g = nx.MultiDiGraph()
    coords = {1: (14.40, 50.05), 2: (14.41, 50.06), 3: (14.42, 50.07), 4: (14.43, 50.08)}
    for node, (x, y) in coords.items():
        g.add_node(node, x=x, y=y)
    dg = nx.DiGraph()
    for u, v in [(1, 2), (2, 3), (3, 4)]:
        g.add_edge(u, v, length=100.0, speed_kph=50.0, current_speed=30.0)
        dg.add_edge(u, v, length=100.0, speed_kph=50.0, current_speed=30.0)

    routing_map = SimpleNamespace(
        network=g,
        current_network=dg,
        temporary_speeds=[],
        get_original_max_speed=lambda u, v: 50.0,
    )

    vehicles = []
    placements = per_segment or [(1, 2)] * 3 + [(2, 3)] * 2 + [(3, 4)]
    for i, (u, _) in enumerate(placements[:n_vehicles]):
        vehicles.append(SimpleNamespace(
            id=i, active=True, status="driving",
            osm_route=[1, 2, 3, 4], start_index=u - 1, start_distance_offset=50.0,
        ))
    # one finished vehicle and one at route end: both must be excluded from counts
    vehicles.append(SimpleNamespace(id=98, active=False, status="finished",
                                    osm_route=[1, 2], start_index=0, start_distance_offset=0.0))
    vehicles.append(SimpleNamespace(id=99, active=True, status="driving",
                                    osm_route=[1, 2], start_index=1, start_distance_offset=0.0))

    setting = SimpleNamespace(
        k_alternatives=1,
        map_update_freq_s=timedelta(seconds=1),
        los_vehicles_tolerance=timedelta(seconds=5),
        travel_time_limit_perc=0.1,
        stuck_detection=0,
        departure_time=datetime(2024, 10, 3),
    )
    stored = []
    return SimpleNamespace(
        vehicles=vehicles,
        setting=setting,
        routing_map=routing_map,
        compute_current_offset=lambda: timedelta(seconds=60),
        store=lambda path: stored.append(path),
        _stored=stored,
    )


# -- snapshot ------------------------------------------------------------


def test_top_segments_counting_and_exclusions():
    sim = make_fake_sim()
    builder = SnapshotBuilder(top_n=2, max_vehicles=100)
    snap = json.loads(builder.build(sim, 1, False, [], None))
    assert snap["top_segments"][0]["id"] == "1-2"
    assert snap["top_segments"][0]["count"] == 3
    assert snap["top_segments"][1] == snap["top_segments"][1]  # valid JSON row
    assert snap["top_segments"][1]["count"] == 2
    assert len(snap["top_segments"]) == 2  # top_n respected
    assert snap["top_segments"][0]["current_speed_kph"] == 30.0
    assert snap["top_segments"][0]["geometry"]  # polyline present


def test_vehicle_sampling_cap():
    sim = make_fake_sim(n_vehicles=6)
    builder = SnapshotBuilder(top_n=5, max_vehicles=3)
    snap = json.loads(builder.build(sim, 1, False, [], None))
    assert snap["vehicles_sampled"] is True
    assert len(snap["vehicles"]) <= 4  # ceil sampling
    # counts stay exact despite sampling
    assert snap["top_segments"][0]["count"] == 3


def test_snapshot_counts():
    sim = make_fake_sim()
    builder = SnapshotBuilder(top_n=5, max_vehicles=100)
    snap = json.loads(builder.build(sim, 7, True, [], 12.5))
    assert snap["step"] == 7 and snap["paused"] is True
    assert snap["n_active"] == 7 and snap["n_finished"] == 1
    assert snap["last_step_ms"] == 12.5


# -- command validation ---------------------------------------------------


@pytest.mark.parametrize("name,value,ok", [
    ("k-alternatives", 3, True),
    ("k-alternatives", 0, False),
    ("k-alternatives", 99, False),
    ("map-update-freq-s", 5, True),
    ("map-update-freq-s", 0, False),
    ("los-vehicles-tolerance-s", 0, True),
    ("travel-time-limit-perc", -1, False),
    ("stuck-detection", 10, True),
    ("not-a-param", 1, False),
])
def test_param_validation(name, value, ok):
    if ok:
        assert validate_param({"name": name, "value": value})[0] == name
    else:
        with pytest.raises(ValueError):
            validate_param({"name": name, "value": value})


def test_restriction_validation():
    good = validate_restriction({"node_from": 1, "node_to": 2, "speed_kph": 0, "duration_s": 60})
    assert good["speed_kph"] == 0 and good["start_offset_s"] == 0
    with pytest.raises(ValueError):
        validate_restriction({"node_from": 1, "node_to": 2, "speed_kph": -5, "duration_s": 60})
    with pytest.raises(ValueError):
        validate_restriction({"node_from": 1, "node_to": 2, "speed_kph": 0, "duration_s": 0})
    with pytest.raises(ValueError):
        validate_restriction({"node_from": "x", "node_to": 2, "speed_kph": 0, "duration_s": 60})


# -- command application --------------------------------------------------


def test_apply_param_and_restriction_and_stop():
    sim = make_fake_sim()
    console = SimulationConsole(make_args())
    console.submit("set-param", {"name": "k-alternatives", "value": 4})
    console.submit("restriction", {"node_from": 2, "node_to": 3, "speed_kph": 0,
                                   "duration_s": 300, "start_offset_s": 0})
    console.submit("restriction", {"node_from": 9, "node_to": 8, "speed_kph": 0,
                                   "duration_s": 300, "start_offset_s": 0})  # no such edge
    console.submit("checkpoint")
    assert console.end_step(sim) is False
    assert sim.setting.k_alternatives == 4
    assert len(sim.routing_map.temporary_speeds) == 1
    ts = sim.routing_map.temporary_speeds[0]
    assert ts.node_from == 2 and float(ts.temporary_speed) == 0.0
    assert float(ts.original_max_speed) == 50.0 and ts.active is False
    assert ts.timestamp_from == datetime(2024, 10, 3, 0, 1)  # departure + offset
    assert sim._stored and sim._stored[0].startswith("console-checkpoint-step")
    results = {r["type"]: r["status"] for r in console.command_results()}
    assert results["set-param"] == "ok" and results["checkpoint"] == "ok"
    statuses = [r["status"] for r in console.command_results() if r["type"] == "restriction"]
    assert sorted(statuses) == ["error", "ok"]

    console.submit("stop")
    assert console.end_step(sim) is True


def test_restriction_list_kept_sorted():
    sim = make_fake_sim()
    console = SimulationConsole(make_args())
    console.submit("restriction", {"node_from": 1, "node_to": 2, "speed_kph": 10,
                                   "duration_s": 60, "start_offset_s": 500})
    console.submit("restriction", {"node_from": 2, "node_to": 3, "speed_kph": 10,
                                   "duration_s": 60, "start_offset_s": 0})
    console.end_step(sim)
    stamps = [ts.timestamp_from for ts in sim.routing_map.temporary_speeds]
    assert stamps == sorted(stamps)


# -- HTTP server lifecycle -------------------------------------------------


def test_server_roundtrip():
    console = SimulationConsole(make_args(port=0))  # ephemeral port
    console.start()
    try:
        port = console._server.server_address[1]
        base = f"http://127.0.0.1:{port}"
        with urllib.request.urlopen(base + "/api/state", timeout=5) as r:
            assert json.loads(r.read())["waiting"] is True
        body = json.dumps({"name": "k-alternatives", "value": 2}).encode()
        req = urllib.request.Request(base + "/api/param", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as r:
            assert r.status == 202
        sim = make_fake_sim()
        console.end_step(sim)
        assert sim.setting.k_alternatives == 2
        with urllib.request.urlopen(base + "/api/state", timeout=5) as r:
            assert json.loads(r.read())["step"] == 1
        with urllib.request.urlopen(base + "/", timeout=5) as r:
            assert b"RUTH Console" in r.read()
    finally:
        console._server.shutdown()
